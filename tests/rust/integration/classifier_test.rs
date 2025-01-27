use prefrontal::{Classifier, BuiltinModel, ClassDefinition};

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
fn test_validation_errors() -> Result<(), Box<dyn std::error::Error>> {
    let result = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(
            ClassDefinition::new(
                "",
                "Empty class name"
            ).with_examples(vec!["test"])
        );
    assert!(result.is_err());

    let result = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(
            ClassDefinition::new(
                "test",
                "Test class"
            ).with_examples(vec![""])
        );
    assert!(result.is_err());

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