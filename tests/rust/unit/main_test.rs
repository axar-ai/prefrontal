use prefrontal::{Classifier, BuiltinModel, ClassDefinition};

fn setup_classifier() -> Classifier {
    Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .unwrap()
        .add_class(
            ClassDefinition::new(
                "sports",
                "Sports and athletic activities"
            ).with_examples(vec!["football game", "basketball match"])
        )
        .unwrap()
        .build()
        .expect("Failed to create classifier")
}

#[test]
fn test_demo_with_different_input() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = setup_classifier();
    let (class, scores) = classifier.predict("chess tournament")?;
    assert_eq!(class, "sports");  // Should still classify as sports due to semantic similarity
    assert!(scores["sports"] > 0.0);
    Ok(())
} 