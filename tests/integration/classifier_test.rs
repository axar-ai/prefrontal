use text_classifier::{Classifier, BuiltinModel, ClassDefinition};
use std::path::PathBuf;

fn get_test_paths() -> (PathBuf, PathBuf) {
    let model_path = PathBuf::from("models/onnx-minilm/model.onnx");
    let tokenizer_path = PathBuf::from("models/onnx-minilm/tokenizer.json");
    (model_path, tokenizer_path)
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
fn test_custom_model_paths() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, tokenizer_path) = get_test_paths();
    let classifier = Classifier::builder()
        .with_custom_model(
            model_path.to_str().unwrap(),
            tokenizer_path.to_str().unwrap(),
            None
        )?
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
fn test_multiple_classes_custom_paths() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, tokenizer_path) = get_test_paths();
    let classifier = Classifier::builder()
        .with_custom_model(
            model_path.to_str().unwrap(),
            tokenizer_path.to_str().unwrap(),
            None
        )?
        .add_class(
            ClassDefinition::new(
                "sports",
                "Sports and athletic activities"
            ).with_examples(vec!["football game", "basketball match"])
        )?
        .add_class(
            ClassDefinition::new(
                "tech",
                "Technology and software"
            ).with_examples(vec!["computer program", "software code"])
        )?
        .build()?;

    let (class_sports, scores_sports) = classifier.predict("soccer match")?;
    let (class_tech, scores_tech) = classifier.predict("python programming")?;

    assert_eq!(class_sports, "sports");
    assert_eq!(class_tech, "tech");
    assert!(scores_sports["sports"] > scores_sports["tech"]);
    assert!(scores_tech["tech"] > scores_tech["sports"]);
    Ok(())
}

#[test]
fn test_validation_errors() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = Classifier::builder().with_model(BuiltinModel::MiniLM)?;
    
    // Test empty label
    assert!(classifier
        .clone()
        .add_class(ClassDefinition::new("", "Empty label test"))
        .is_err());
    
    // Test empty description
    assert!(classifier
        .clone()
        .add_class(ClassDefinition::new("label", ""))
        .is_err());
    
    // Test empty example
    assert!(classifier
        .clone()
        .add_class(
            ClassDefinition::new("label", "Test")
                .with_examples(vec![""])
        )
        .is_err());
    
    Ok(())
}

#[test]
fn test_zero_shot_classification() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(
            ClassDefinition::new(
                "sports",
                "Content about sports, athletics, and competitive games"
            )
        )?
        .add_class(
            ClassDefinition::new(
                "tech",
                "Content about technology, computers, and software"
            )
        )?
        .build()?;

    let (class, scores) = classifier.predict("The new programming language features")?;
    assert_eq!(class, "tech");
    assert!(scores["tech"] > scores["sports"]);
    
    Ok(())
}

#[test]
fn test_same_results_builtin_and_custom() -> Result<(), Box<dyn std::error::Error>> {
    // Create two classifiers - one with built-in model and one with custom paths
    let builtin = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(
            ClassDefinition::new(
                "sports",
                "Sports and athletic activities"
            ).with_examples(vec!["football game", "basketball match"])
        )?
        .build()?;

    let (model_path, tokenizer_path) = get_test_paths();
    let custom = Classifier::builder()
        .with_custom_model(
            model_path.to_str().unwrap(),
            tokenizer_path.to_str().unwrap(),
            None
        )?
        .add_class(
            ClassDefinition::new(
                "sports",
                "Sports and athletic activities"
            ).with_examples(vec!["football game", "basketball match"])
        )?
        .build()?;

    // Make predictions with both
    let (class1, scores1) = builtin.predict("soccer match")?;
    let (class2, scores2) = custom.predict("soccer match")?;

    // Verify they give same results
    assert_eq!(class1, class2);
    assert_eq!(scores1["sports"], scores2["sports"]);
    Ok(())
} 