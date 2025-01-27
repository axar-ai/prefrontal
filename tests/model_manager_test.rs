use prefrontal::{ModelManager, ModelError, BuiltinModel, ModelInfo};

#[tokio::test]
async fn test_model_download() -> Result<(), Box<dyn std::error::Error>> {
    let manager = ModelManager::new_default()?;

    let info = ModelInfo {
        name: "minilm".to_string(),
        model_url: "https://huggingface.co/axar-ai/minilm/resolve/main/model.onnx".to_string(),
        tokenizer_url: "https://huggingface.co/axar-ai/minilm/resolve/main/tokenizer.json".to_string(),
        model_hash: "37f1ea074b7166e87295fce31299287d5fb79f76b8b7227fccc8a9f2f1ba4e16".to_string(),
        tokenizer_hash: "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0".to_string(),
    };

    // Clean up any existing files
    let model_path = manager.get_model_path(&info.name);
    let tokenizer_path = manager.get_tokenizer_path(&info.name);
    if model_path.exists() {
        std::fs::remove_file(&model_path)?;
    }
    if tokenizer_path.exists() {
        std::fs::remove_file(&tokenizer_path)?;
    }

    assert!(!manager.is_model_downloaded("minilm"));
    
    manager.download_model(&info).await?;
    assert!(manager.is_model_downloaded("minilm"));
    assert!(manager.verify_model(&info)?);

    Ok(())
}

#[tokio::test]
async fn test_model_verification() -> Result<(), Box<dyn std::error::Error>> {
    let manager = ModelManager::new_default()?;

    let info = ModelInfo {
        name: "minilm".to_string(),
        model_url: "https://huggingface.co/axar-ai/minilm/resolve/main/model.onnx".to_string(),
        tokenizer_url: "https://huggingface.co/axar-ai/minilm/resolve/main/tokenizer.json".to_string(),
        model_hash: "37f1ea074b7166e87295fce31299287d5fb79f76b8b7227fccc8a9f2f1ba4e16".to_string(),
        tokenizer_hash: "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0".to_string(),
    };

    // Clean up any existing files
    let model_path = manager.get_model_path(&info.name);
    let tokenizer_path = manager.get_tokenizer_path(&info.name);
    if model_path.exists() {
        std::fs::remove_file(&model_path)?;
    }
    if tokenizer_path.exists() {
        std::fs::remove_file(&tokenizer_path)?;
    }

    assert!(!manager.verify_model(&info)?);
    Ok(())
}

#[tokio::test]
async fn test_model_paths() -> Result<(), Box<dyn std::error::Error>> {
    let manager = ModelManager::new_default()?;
    let model_path = manager.get_model_path("minilm");
    let tokenizer_path = manager.get_tokenizer_path("minilm");

    assert!(model_path.ends_with("minilm/model.onnx"));
    assert!(tokenizer_path.ends_with("minilm/tokenizer.json"));
    Ok(())
}

#[tokio::test]
async fn test_default_model_manager() -> Result<(), ModelError> {
    let manager = ModelManager::new_default()?;
    let model_info = BuiltinModel::MiniLM.get_model_info();

    // Clean up any existing files
    let model_path = manager.get_model_path(&model_info.name);
    let tokenizer_path = manager.get_tokenizer_path(&model_info.name);
    if model_path.exists() {
        std::fs::remove_file(&model_path)?;
    }
    if tokenizer_path.exists() {
        std::fs::remove_file(&tokenizer_path)?;
    }

    // Test download to default location
    assert!(!manager.is_model_downloaded(&model_info.name));
    manager.download_model(&model_info).await?;

    assert!(manager.is_model_downloaded(&model_info.name));
    assert!(manager.verify_model(&model_info)?);

    Ok(())
} 