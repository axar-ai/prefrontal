use prefrontal::{Classifier, BuiltinModel, ClassDefinition, ClassifierError};

#[tokio::test]
async fn test_validation() -> Result<(), Box<dyn std::error::Error>> {
    let model = BuiltinModel::MiniLM;

    // Test empty class
    let result = Classifier::builder()
        .with_model(model)?
        .add_class(
            ClassDefinition::new("empty", "Empty class")
                .with_examples(vec![""])
        );
    assert!(result.is_err());

    // Test invalid class label
    let result = Classifier::builder()
        .with_model(model)?
        .add_class(ClassDefinition::new("", "Empty label"));
    assert!(result.is_err());

    // Test invalid description
    let result = Classifier::builder()
        .with_model(model)?
        .add_class(ClassDefinition::new("label", ""));
    assert!(result.is_err());

    Ok(())
}

#[tokio::test]
async fn test_many_classes() -> Result<(), ClassifierError> {
    let model = BuiltinModel::MiniLM;
    let mut builder = Classifier::builder().with_model(model)?;

    // Add maximum number of classes
    for i in 0..100 {
        builder = builder.add_class(
            ClassDefinition::new(
                &format!("class{}", i),
                &format!("Description for class {}", i)
            ).with_examples(vec!["example"])
        )?;
    }

    // Try to add one more class
    let result = builder.add_class(
        ClassDefinition::new(
            "one_too_many",
            "This class should exceed the limit"
        ).with_examples(vec!["example"])
    );

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ClassifierError::ValidationError(_)));

    Ok(())
}

#[tokio::test]
async fn test_complex_end_to_end() -> Result<(), ClassifierError> {
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(
            ClassDefinition::new("sports", "Sports content")
                .with_examples(vec!["football", "basketball", "tennis"])
        )?
        .add_class(
            ClassDefinition::new("tech", "Technology content")
                .with_examples(vec!["computer", "software", "programming"])
        )?
        .add_class(
            ClassDefinition::new("music", "Music content")
                .with_examples(vec!["guitar", "piano", "concert"])
        )?
        .build()?;

    let test_texts = vec![
        "The new programming language features are amazing",
        "The basketball game was intense",
        "The concert last night was incredible",
    ];

    for text in test_texts {
        let (label, scores) = classifier.predict(text)?;
        assert!(!label.is_empty());
        assert!(scores.values().all(|&score| score >= 0.0 && score <= 1.0));
    }

    Ok(())
} 