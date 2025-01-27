use std::fs;
use prefrontal::{ModelManager, BuiltinModel};
use std::path::PathBuf;

fn get_test_dir(test_name: &str) -> PathBuf {
    PathBuf::from("/tmp")
        .join("test-prefrontal")
        .join("models")
        .join(test_name)
}

fn clean_test_dir(test_name: &str) -> PathBuf {
    let test_dir = get_test_dir(test_name);
    let _ = fs::remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).unwrap();
    test_dir
}

/// Ensures the model exists in the given directory, downloading only if necessary.
/// Returns true if the model was downloaded, false if it was already present.
pub async fn ensure_model_exists(manager: &ModelManager, model: BuiltinModel) -> Result<bool, Box<dyn std::error::Error>> {
    if manager.is_model_downloaded(model) && manager.verify_model(model)? {
        Ok(false) // Model already exists and is valid
    } else {
        manager.download_model(model).await?;
        Ok(true) // Model was downloaded
    }
}

#[tokio::test]
async fn test_model_download_and_verify() -> Result<(), Box<dyn std::error::Error>> {
    // This test needs a clean directory to verify download behavior
    let test_dir = clean_test_dir("download_and_verify");
    let manager = ModelManager::new(&test_dir).unwrap();
    let model = BuiltinModel::MiniLM;

    // Test download
    assert!(!manager.is_model_downloaded(model));
    manager.download_model(model).await?;
    assert!(manager.is_model_downloaded(model));
    assert!(manager.verify_model(model)?);

    Ok(())
}

#[tokio::test]
async fn test_model_paths() -> Result<(), Box<dyn std::error::Error>> {
    // This test only checks path construction, no need to clean or download
    let test_dir = get_test_dir("model_paths");
    let manager = ModelManager::new(&test_dir).unwrap();
    let model = BuiltinModel::MiniLM;

    let model_path = manager.get_model_path(model);
    let tokenizer_path = manager.get_tokenizer_path(model);

    assert!(model_path.to_str().unwrap().contains("model.onnx"));
    assert!(tokenizer_path.to_str().unwrap().contains("tokenizer.json"));

    Ok(())
}

#[tokio::test]
async fn test_model_verification() -> Result<(), Box<dyn std::error::Error>> {
    // This test needs a clean directory to test verification states
    let test_dir = clean_test_dir("model_verification");
    let manager = ModelManager::new(&test_dir).unwrap();
    let model = BuiltinModel::MiniLM;

    // Test verification of non-existent model
    assert!(!manager.verify_model(model)?);

    // Download and verify
    manager.download_model(model).await?;
    assert!(manager.verify_model(model)?);

    // Corrupt file and verify
    fs::write(manager.get_model_path(model), "corrupted data")?;
    assert!(!manager.verify_model(model)?);

    Ok(())
}

#[tokio::test]
async fn test_ensure_model_downloaded() -> Result<(), Box<dyn std::error::Error>> {
    // This test needs a clean directory to test ensure_model_downloaded behavior
    let test_dir = clean_test_dir("ensure_downloaded");
    let manager = ModelManager::new(&test_dir).unwrap();
    let model = BuiltinModel::MiniLM;

    // Test initial download
    assert!(!manager.is_model_downloaded(model));
    manager.ensure_model_downloaded(model).await?;
    assert!(manager.is_model_downloaded(model));

    // Test verification and re-download
    fs::write(manager.get_model_path(model), "corrupted data")?;
    manager.ensure_model_downloaded(model).await?;
    assert!(manager.verify_model(model)?);

    Ok(())
}

#[tokio::test]
async fn test_model_info() -> Result<(), Box<dyn std::error::Error>> {
    // This test only checks model info, no need to clean or download
    let model = BuiltinModel::MiniLM;
    let info = model.get_model_info();

    assert_eq!(info.name, "minilm");
    assert!(info.model_url.contains("model.onnx"));
    assert!(info.tokenizer_url.contains("tokenizer.json"));
    assert!(!info.model_hash.is_empty());
    assert!(!info.tokenizer_hash.is_empty());

    Ok(())
} 