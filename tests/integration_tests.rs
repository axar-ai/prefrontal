use text_classifier::{Classifier, ClassifierError, ClassDefinition, BuiltinModel};

fn get_test_paths() -> (BuiltinModel, String) {
    let tokenizer_path = "models/bert-base-uncased".to_string();
    (BuiltinModel::MiniLM, tokenizer_path)
}

#[test]
fn test_basic_validation() -> Result<(), ClassifierError> {
    let (model, _) = get_test_paths();
    
    let result = Classifier::builder()
        .with_model(model)?
        .add_class(
            ClassDefinition::new("test", "Test class")
                .with_examples(vec!["sample text"])
        )?
        .build();

    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_end_to_end() -> Result<(), ClassifierError> {
    let (model, _) = get_test_paths();
    
    let classifier = Classifier::builder()
        .with_model(model)?
        .add_class(
            ClassDefinition::new("tech", "Technology related content")
                .with_examples(vec![
                    "artificial intelligence breakthrough",
                    "new software release",
                    "quantum computing advance",
                ])
        )?
        .add_class(
            ClassDefinition::new("sports", "Sports related content")
                .with_examples(vec![
                    "championship game victory",
                    "world record in athletics",
                    "tournament final match",
                ])
        )?
        .add_class(
            ClassDefinition::new("business", "Business and finance content")
                .with_examples(vec![
                    "stock market analysis",
                    "corporate merger announcement",
                    "startup funding round",
                ])
        )?
        .build()?;

    let (class, scores) = classifier.predict("The startup announced a major AI breakthrough")?;
    assert!(!scores.is_empty());
    assert_eq!(class, "tech"); // The text is about AI, so it should be classified as tech
    Ok(())
}

#[test]
fn test_duplicate_class() {
    let (model, _) = get_test_paths();
    
    let result = Classifier::builder()
        .with_model(model)
        .and_then(|builder| {
            builder.add_class(
                ClassDefinition::new("test", "Test class 1")
                    .with_examples(vec!["example1"])
            )
        })
        .and_then(|builder| {
            builder.add_class(
                ClassDefinition::new("test", "Test class 2")
                    .with_examples(vec!["example2"])
            )
        });

    assert!(result.is_err());
}

#[test]
fn test_empty_class_label() {
    let (model, _) = get_test_paths();
    
    let result = Classifier::builder()
        .with_model(model)
        .and_then(|builder| {
            builder.add_class(
                ClassDefinition::new("", "Empty label")
                    .with_examples(vec!["example"])
            )
        });

    assert!(result.is_err());
}

#[test]
fn test_empty_example() {
    let (model, _) = get_test_paths();
    
    let result = Classifier::builder()
        .with_model(model)
        .and_then(|builder| {
            builder.add_class(
                ClassDefinition::new("test", "Test class")
                    .with_examples(vec![""])
            )
        });

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ClassifierError::ValidationError(_)));
}

#[test]
fn test_many_classes() -> Result<(), ClassifierError> {
    let (model, _) = get_test_paths();
    let mut builder = Classifier::builder()
        .with_model(model)?;

    for i in 0..10 {
        builder = builder.add_class(
            ClassDefinition::new(
                format!("class_{}", i),
                format!("Test class {}", i)
            ).with_examples(vec!["example text"])
        )?;
    }

    let classifier = builder.build()?;
    let (class, scores) = classifier.predict("test prediction")?;
    assert!(!scores.is_empty());
    assert!(class.starts_with("class_")); // Should match one of our classes
    Ok(())
}

#[test]
fn test_long_description() {
    let (model, _) = get_test_paths();
    
    // Create a description that's too long (1001 characters)
    let long_description = "a".repeat(1001);
    
    let result = Classifier::builder()
        .with_model(model)
        .and_then(|builder| {
            builder.add_class(
                ClassDefinition::new("test", long_description)
                    .with_examples(vec!["example"])
            )
        });

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ClassifierError::ValidationError(_)));
} 