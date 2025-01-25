use text_classifier::Classifier;
use std::path::PathBuf;

fn get_test_paths() -> (PathBuf, PathBuf) {
    let model_path = PathBuf::from("models/onnx-minilm/model.onnx");
    let tokenizer_path = PathBuf::from("models/onnx-minilm/tokenizer.json");
    (model_path, tokenizer_path)
}

#[test]
fn test_end_to_end_classification() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, tokenizer_path) = get_test_paths();
    let mut classifier = Classifier::new(
        model_path.to_str().unwrap(),
        tokenizer_path.to_str().unwrap()
    )?;

    classifier.add_class("sports", vec!["football game", "basketball match"])?;
    classifier.build()?;

    let (class, scores) = classifier.predict("soccer match")?;
    
    assert_eq!(class, "sports");
    assert!(scores.contains_key("sports"));
    assert!(scores["sports"] > 0.5);
    Ok(())
}

#[test]
fn test_multiple_classes() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, tokenizer_path) = get_test_paths();
    let mut classifier = Classifier::new(
        model_path.to_str().unwrap(),
        tokenizer_path.to_str().unwrap()
    )?;

    classifier.add_class("sports", vec!["football game", "basketball match"])?;
    classifier.add_class("tech", vec!["computer program", "software code"])?;
    classifier.build()?;

    let (class_sports, scores_sports) = classifier.predict("soccer match")?;
    let (class_tech, scores_tech) = classifier.predict("python programming")?;

    assert_eq!(class_sports, "sports");
    assert_eq!(class_tech, "tech");
    assert!(scores_sports["sports"] > scores_sports["tech"]);
    assert!(scores_tech["tech"] > scores_tech["sports"]);
    Ok(())
}

#[test]
fn test_validation_errors() {
    assert!(Classifier::new("", "tokenizer.json").is_err());
    assert!(Classifier::new("model.onnx", "").is_err());
    
    let (model_path, tokenizer_path) = get_test_paths();
    let mut classifier = Classifier::new(
        model_path.to_str().unwrap(),
        tokenizer_path.to_str().unwrap()
    ).expect("Failed to create classifier");
    
    assert!(classifier.add_class("", vec!["example"]).is_err());
    assert!(classifier.add_class("label", vec![]).is_err());
    assert!(classifier.add_class("label", vec![""]).is_err());
} 