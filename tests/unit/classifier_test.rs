use text_classifier::{Classifier, BuiltinModel, ClassifierError};
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
fn test_classifier_thread_safety() {
    let classifier = setup_test_classifier();
    let classifier = Arc::new(classifier);
    let mut handles = vec![];
    
    // Spawn multiple threads that use the classifier concurrently
    for i in 0..3 {
        let classifier = Arc::clone(&classifier);
        let handle = thread::spawn(move || {
            let text = match i {
                0 => "test example",
                1 => "another test",
                _ => "final test",
            };
            classifier.predict(text).unwrap()
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
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