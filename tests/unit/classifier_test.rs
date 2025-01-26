use text_classifier::{Classifier, BuiltinModel, ClassDefinition};
use ndarray::Array1;
use std::sync::Arc;
use std::thread;

fn setup_test_classifier() -> Classifier {
    Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .unwrap()
        .add_class("test", vec!["example text"])
        .unwrap()
        .build()
        .expect("Failed to create classifier")
}

#[test]
fn test_empty_class_handling() {
    let result = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .unwrap()
        .add_class("empty", vec![])
        .unwrap()
        .build();
    assert!(result.is_err());
}

#[test]
fn test_unknown_class_prediction() {
    let classifier = setup_test_classifier();
    let result = classifier.predict("test");
    assert!(result.is_ok());
    let (class, scores) = result.unwrap();
    assert_eq!(class, "test"); // Since we added "test" class in setup
    assert!(!scores.is_empty());
}

#[test]
fn test_tokenizer_padding() {
    let classifier = setup_test_classifier();
    let result = classifier.predict("short text");
    assert!(result.is_ok());
}

#[test]
fn test_token_length_validation() {
    let classifier = setup_test_classifier();
    let very_long_text = "a ".repeat(1000);
    let result = classifier.predict(&very_long_text);
    assert!(matches!(result, Err(ClassifierError::ValidationError(_))));
}

#[test]
fn test_empty_input_validation() {
    let classifier = setup_test_classifier();
    let result = classifier.predict("");
    assert!(matches!(result, Err(ClassifierError::ValidationError(_))));
}

#[test]
fn test_token_counting() {
    let classifier = setup_test_classifier();
    let result = classifier.count_tokens("test text");
    assert!(result.is_ok());
    assert!(result.unwrap() > 0);
}

#[test]
fn test_build_idempotent() {
    let mut classifier = setup_test_classifier();
    classifier.add_class("test", vec!["example"]);
    
    let first_build = classifier.build();
    let second_build = classifier.build();
    
    assert!(first_build.is_ok());
    assert!(second_build.is_ok());
}

#[test]
fn test_normalize_vector() {
    let vec = Array1::from_vec(vec![3.0, 4.0]);  // 3-4-5 triangle
    let normalized = Classifier::normalize_vector(&vec);
    
    assert!((normalized[0] - 0.6).abs() < 1e-6);  // 3/5
    assert!((normalized[1] - 0.8).abs() < 1e-6);  // 4/5
}

#[test]
fn test_thread_safety() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = Arc::new(
        Classifier::builder()
            .with_model(BuiltinModel::MiniLM)?
            .add_class(
                ClassDefinition::new(
                    "sports",
                    "Sports and athletic activities"
                ).with_examples(vec!["football game", "basketball match"])
            )?
            .build()?
    );

    let mut handles = vec![];
    
    for _ in 0..4 {
        let classifier_clone = Arc::clone(&classifier);
        let handle = thread::spawn(move || {
            let (class, scores) = classifier_clone.predict("soccer match").unwrap();
            assert_eq!(class, "sports");
            assert!(scores["sports"] > 0.5);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    Ok(())
}

#[test]
fn test_class_validation() -> Result<(), Box<dyn std::error::Error>> {
    // Test invalid class label
    assert!(Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(ClassDefinition::new("", "Empty label"))
        .is_err());
    
    // Test invalid description
    assert!(Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(ClassDefinition::new("label", ""))
        .is_err());
    
    // Test invalid examples
    assert!(Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(
            ClassDefinition::new("label", "Test class")
                .with_examples(vec![""])
        )
        .is_err());
    
    Ok(())
}

#[test]
fn test_prediction_validation() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(
            ClassDefinition::new(
                "sports",
                "Sports and athletic activities"
            ).with_examples(vec!["football game"])
        )?
        .build()?;

    // Test empty input
    assert!(classifier.predict("").is_err());
    
    // Test very long input (should be truncated internally)
    let long_text = "a".repeat(1000);
    let (class, _) = classifier.predict(&long_text)?;
    assert_eq!(class, "sports"); // Should still work with truncation
    
    Ok(())
}

#[test]
fn test_zero_shot_predictions() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(
            ClassDefinition::new(
                "sports",
                "Content about sports and athletics"
            )
        )?
        .add_class(
            ClassDefinition::new(
                "tech",
                "Content about technology and computers"
            )
        )?
        .build()?;

    let (class, scores) = classifier.predict("The latest software update")?;
    assert_eq!(class, "tech");
    assert!(scores["tech"] > scores["sports"]);

    let (class, scores) = classifier.predict("Championship game highlights")?;
    assert_eq!(class, "sports");
    assert!(scores["sports"] > scores["tech"]);

    Ok(())
}

#[test]
fn test_classifier_clone_and_send() {
    let classifier = setup_test_classifier();
    
    // Test that classifier can be sent to another thread
    thread::spawn(move || {
        classifier.predict("test").unwrap();
    }).join().unwrap();
}

// ... rest of the unit tests from src/classifier.rs ... 