use prefrontal::{Classifier, BuiltinModel, ClassDefinition};
use std::sync::Arc;
use std::thread;

fn setup_test_classifier() -> Classifier {
    Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .unwrap()
        .add_class(
            ClassDefinition::new("test", "Test class")
                .with_examples(vec!["example text"])
        )
        .unwrap()
        .build()
        .expect("Failed to create classifier")
}

#[test]
fn test_end_to_end_classification() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(
            ClassDefinition::new(
                "sports",
                "Sports and athletic activities"
            ).with_examples(vec!["football game", "basketball match"])
        )?
        .build()?;

    let (class, scores) = classifier.predict("soccer match")?;
    
    assert_eq!(class, "sports");
    assert!(scores.contains_key("sports"));
    assert!(scores["sports"] > 0.5);
    Ok(())
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
fn test_token_length_handling() {
    let classifier = setup_test_classifier();
    // Create a text that will definitely exceed the max length (128 tokens)
    let very_long_text = "this is a test sentence with multiple words that should be tokenized into individual tokens and hopefully exceed the maximum length limit of the tokenizer which is set to 128 tokens in the configuration file and we need to make sure this text is long enough to trigger that validation error so we will keep adding more words until we reach that limit and here are some more words to make it even longer because apparently the tokenizer is quite efficient at handling long texts and we need more words to exceed the limit ".repeat(20);
    
    // First verify that the text is indeed longer than 128 tokens
    let token_count = classifier.count_tokens(&very_long_text).unwrap();
    assert_eq!(token_count, 128, "Expected tokenizer to truncate at 128 tokens");
    
    // Verify that prediction still works with truncated input
    let result = classifier.predict(&very_long_text);
    assert!(result.is_ok(), "Prediction should succeed with truncated input");
}

#[test]
fn test_thread_safety() {
    let classifier = Arc::new(setup_test_classifier());
    let mut handles = vec![];

    for _ in 0..3 {
        let classifier = Arc::clone(&classifier);
        let handle = thread::spawn(move || {
            let result = classifier.predict("test text");
            assert!(result.is_ok());
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
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
fn test_zero_shot_classification() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(
            ClassDefinition::new(
                "sports",
                "Content about sports and athletic activities"
            ).with_examples(vec!["football game", "basketball match"])
        )?
        .add_class(
            ClassDefinition::new(
                "tech",
                "Content about technology, programming, and computers"
            ).with_examples(vec!["python code", "machine learning"])
        )?
        .build()?;

    let (class, scores) = classifier.predict("how to code in rust")?;
    assert_eq!(class, "tech");
    assert!(scores["tech"] > scores["sports"]);
    Ok(())
}

#[test]
fn test_classifier_thread_safety() {
    let classifier = setup_test_classifier();
    
    // Test that classifier can be sent to another thread
    thread::spawn(move || {
        classifier.predict("test").unwrap();
    }).join().unwrap();
}

#[test]
fn test_multiple_classes() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(
            ClassDefinition::new(
                "sports",
                "Sports and athletic activities"
            ).with_examples(vec!["football game", "basketball match"])
        )?
        .add_class(
            ClassDefinition::new(
                "tech",
                "Technology and programming"
            ).with_examples(vec!["python code", "machine learning"])
        )?
        .build()?;

    let (class, scores) = classifier.predict("javascript programming")?;
    assert_eq!(class, "tech");
    assert!(scores["tech"] > scores["sports"]);
    Ok(())
}

#[test]
fn test_semantic_similarity() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(
            ClassDefinition::new(
                "sports",
                "Sports and athletic activities"
            ).with_examples(vec!["football game", "basketball match"])
        )?
        .build()?;

    let (class, scores) = classifier.predict("chess tournament")?;
    assert_eq!(class, "sports");  // Should classify as sports due to semantic similarity
    assert!(scores["sports"] > 0.0);
    Ok(())
} 