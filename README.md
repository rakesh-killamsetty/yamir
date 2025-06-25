#CONCEPT-
# Few-Shot Object Detector

This is a minimal project that demonstrates few-shot image object detection using a Siamese Network and sliding window detection. It highlights the most similar object in an image using cosine similarity between embeddings.


 Project Structure

fewshot_detector/
├── data/
│   ├── support1.jpg         # Few-shot support image (1 or few samples)
│   └── query1.jpg           # Test image to detect object in
├── detect.py                # Main detection script
├── models/
│   └── siamese.py           # Siamese network architecture
├── utils/
│   ├── preprocess.py        # Image preprocessing and embedding
│   └── window.py            # Sliding window logic
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation


 Features
- Siamese Network with ResNet-18 backbone
- Cosine similarity for image comparison
- Manual sliding window for region proposals
- Draws bounding box around best match
- Outputs area of largest detected match

