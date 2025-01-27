use criterion::{black_box, criterion_group, criterion_main, Criterion};
use prefrontal::{Classifier, BuiltinModel, ClassDefinition, ModelManager};
use tokio::runtime::Runtime;

async fn setup_model() -> Result<ModelManager, Box<dyn std::error::Error>> {
    let manager = ModelManager::new_default()?;
    let model = BuiltinModel::MiniLM;
    
    if !manager.is_model_downloaded(model) {
        manager.download_model(model).await?;
    }
    assert!(manager.is_model_downloaded(model));
    Ok(manager)
}

async fn setup_classifier() -> Result<Classifier, Box<dyn std::error::Error>> {
    let _manager = setup_model().await?;

    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(
            ClassDefinition::new("positive", "Positive sentiment")
                .with_examples(vec!["great", "awesome", "excellent"])
        )?
        .add_class(
            ClassDefinition::new("negative", "Negative sentiment")
                .with_examples(vec!["terrible", "awful", "bad"])
        )?
        .build()?;

    Ok(classifier)
}

fn bench_embedding(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let classifier = rt.block_on(setup_classifier()).unwrap();
    let mut group = c.benchmark_group("Embedding");
    
    // Short text
    group.bench_function("short_text", |b| b.iter(|| {
        classifier.predict(black_box("This is a short text")).unwrap()
    }));
    
    // Medium text
    group.bench_function("medium_text", |b| b.iter(|| {
        classifier.predict(black_box(
            "This is a medium length text that should take more time to process \
             due to its increased length and complexity"
        )).unwrap()
    }));
    
    // Long text
    group.bench_function("long_text", |b| b.iter(|| {
        classifier.predict(black_box(
            "This is a much longer text that contains multiple sentences and should \
             take significantly more time to process. It includes various words and \
             punctuation marks, making it a good test case for performance. \
             The length of this text should help us understand how the model scales \
             with input size."
        )).unwrap()
    }));
    
    group.finish();
}

fn bench_embedding_generation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let classifier = rt.block_on(setup_classifier()).unwrap();
    let mut group = c.benchmark_group("Embedding");
    
    group.sample_size(50)  // Reduce sample size for long running benchmarks
         .measurement_time(std::time::Duration::from_secs(10));  // Increase measurement time
    
    group.bench_function("single_text", |b| b.iter(|| {
        classifier.predict(black_box("Sample text for embedding benchmark")).unwrap()
    }));
    
    group.finish();
}

fn bench_classification(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Classification");
    
    group.sample_size(50)  // Reduce sample size for long running benchmarks
         .measurement_time(std::time::Duration::from_secs(10));  // Increase measurement time
    
    // Single class classification with timing
    {
        let manager = rt.block_on(setup_model()).unwrap();
        assert!(manager.is_model_downloaded(BuiltinModel::MiniLM));
        
        let classifier = Classifier::builder()
            .with_model(BuiltinModel::MiniLM)
            .unwrap()
            .add_class(
                ClassDefinition::new(
                    "sports",
                    "Sports-related content"
                ).with_examples(vec!["football game", "basketball match"])
            )
            .unwrap()
            .build()
            .unwrap();
        
        group.bench_function("single_prediction", |b| b.iter(|| {
            classifier.predict(black_box("Test text for benchmark")).unwrap()
        }));
    }
    
    // Multiple classes with timing
    {
        let manager = rt.block_on(setup_model()).unwrap();
        assert!(manager.is_model_downloaded(BuiltinModel::MiniLM));
        
        let mut builder = Classifier::builder()
            .with_model(BuiltinModel::MiniLM)
            .unwrap();
            
        for i in 0..10 {
            builder = builder.add_class(
                ClassDefinition::new(
                    &format!("class_{}", i),
                    &format!("Test class {}", i)
                ).with_examples(vec!["example 1", "example 2", "example 3"])
            ).unwrap();
        }
        
        let classifier = builder.build().unwrap();
        
        group.bench_function("prediction_ten_classes", |b| b.iter(|| {
            classifier.predict(black_box("Test text for benchmark")).unwrap()
        }));
    }
    
    group.finish();
}

fn bench_build_time(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let manager = rt.block_on(setup_model()).unwrap();
    assert!(manager.is_model_downloaded(BuiltinModel::MiniLM));
    
    let mut group = c.benchmark_group("Build");
    
    group.sample_size(30)  // Further reduce sample size for very long running benchmarks
         .measurement_time(std::time::Duration::from_secs(15));  // Increase measurement time
    
    group.bench_function("build_ten_classes", |b| b.iter(|| {
        let mut builder = Classifier::builder()
            .with_model(BuiltinModel::MiniLM)
            .unwrap();
            
        for i in 0..10 {
            builder = builder.add_class(
                ClassDefinition::new(
                    &format!("class_{}", i),
                    &format!("Test class {}", i)
                ).with_examples(vec!["example 1", "example 2", "example 3"])
            ).unwrap();
        }
        
        builder.build().unwrap()
    }));
    
    group.finish();
}

criterion_group!(
    benches,
    bench_embedding,
    bench_embedding_generation,
    bench_classification,
    bench_build_time
);
criterion_main!(benches); 