use text_classifier::Classifier;

#[test]
fn test_basic_classification() -> Result<(), Box<dyn std::error::Error>> {
    let mut classifier = Classifier::new(
        "models/onnx-minilm/model.onnx",
        "models/onnx-minilm/tokenizer.json"
    )?;

    classifier.add_class("test", vec!["test example"])?;
    classifier.build()?;
    
    let (class, scores) = classifier.predict("test example")?;
    assert_eq!(class, "test");
    assert!(scores["test"] > 0.5);
    Ok(())
}

#[test]
fn test_multiple_examples() -> Result<(), Box<dyn std::error::Error>> {
    let mut classifier = Classifier::new(
        "models/onnx-minilm/model.onnx",
        "models/onnx-minilm/tokenizer.json"
    )?;

    classifier.add_class("test", vec!["example one", "example two"])?;
    classifier.build()?;

    let (class, scores) = classifier.predict("test example")?;
    assert_eq!(class, "test");
    assert!(scores["test"] > 0.5);
    Ok(())
}

#[test]
fn test_multiple_classes() -> Result<(), Box<dyn std::error::Error>> {
    let mut classifier = Classifier::new(
        "models/onnx-minilm/model.onnx",
        "models/onnx-minilm/tokenizer.json"
    )?;

    classifier.add_class("sports", vec!["football game", "basketball match"])?;
    classifier.add_class("technology", vec!["python coding", "software development"])?;
    classifier.build()?;

    let (class_sports, scores_sports) = classifier.predict("soccer match")?;
    let (class_tech, scores_tech) = classifier.predict("python programming")?;

    assert_eq!(class_sports, "sports");
    assert_eq!(class_tech, "technology");
    assert!(scores_sports["sports"] > scores_sports["technology"]);
    assert!(scores_tech["technology"] > scores_tech["sports"]);
    Ok(())
}

#[test]
fn test_validation_errors() {
    assert!(Classifier::new("", "tokenizer.json").is_err());
    assert!(Classifier::new("model.onnx", "").is_err());
    
    let mut classifier = Classifier::new(
        "models/onnx-minilm/model.onnx",
        "models/onnx-minilm/tokenizer.json"
    ).expect("Failed to create classifier");
    
    assert!(classifier.add_class("", vec!["example"]).is_err());
    assert!(classifier.add_class("label", vec![]).is_err());
    assert!(classifier.add_class("label", vec![""]).is_err());
} 