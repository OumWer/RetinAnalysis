from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns



modelpath = "...\\clip_retina_epoch_10.pt"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# Load the fine-tuned weights
checkpoint = torch.load(modelpath)
model.load_state_dict(torch.load(modelpath))
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

torch.set_printoptions(sci_mode=False)

def load_image_from_url(url):
    try:
        # Get the image from the URL
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors

        # Open the image with PIL
        img = Image.open(BytesIO(response.content))
        return img

    except requests.exceptions.RequestException as e:
        print(f"Error loading image: {e}")
        return None
    
dataset = pd.read_excel("D:\Labeled Retinology Images partie.xlsx")
dataset_labels = dataset['Label'].tolist() 
def display_images(images, num_columns=8):
    """
    Displays a list of images in a grid with a specified number of columns and automatically calculates rows.
    """
    # Calculate the number of rows required based on the number of images and columns
    num_images = len(images)
    num_rows = math.ceil(num_images / num_columns)

    # Create a figure with the specified number of rows and columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))

    # Flatten the axes array to easily iterate over it
    axes = axes.flatten()

    # Loop through images and axes, displaying images in the grid
    for idx, image in enumerate(images):
        ax = axes[idx]
        if image.mode == 'L':
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(image)
        ax.set_title(f"Image {idx + 1}")
        ax.axis("off")  # Hide axes for a cleaner view

    # If there are remaining axes (in case the grid is not completely filled), hide them
    for idx in range(len(images), len(axes)):
        axes[idx].axis("off")

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()



images = []
for url in dataset['ImageURL']:  # Assuming the column with image URLs is named 'imageurl'
    image = load_image_from_url(url)
    if image:
        images.append(image)

display_images(images)

texts = [
    "Dry AMD",
    "Glaucoma",
    "Normal Fundus",
    "Wet AMD",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferate DR",
    "Cataract",
    "Hypertensive Retinopathy",
    "Pathological Myopia"
]


inputs = processor(
    text=texts,
    images=images,
    return_tensors="pt",
    padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

for idx, image in enumerate(images):
    max_prob_idx = torch.argmax(probs[idx]).item()  # Find the index of the text with the maximum probability
    predicted_label = texts[max_prob_idx]  # Get the predicted text (label) based on the highest probability
    actual_label = dataset_labels[idx]  # Get the actual label from the dataset
    
    print(f"Image {idx+1}:")
    print(f"  Actual Label (from dataset): {actual_label}")
    print(f"  Predicted Label (from text with max prob): {predicted_label}")
    print(f"  Max Probability: {probs[idx][max_prob_idx].item()}")
    print("-" * 50)

# Zero shot

# Obtain image embeddings


def compute_similarity(images):
    embeddings = []

    with torch.no_grad():
        for image in images:
            preprocess = processor(images=image, return_tensors="pt")['pixel_values']
            embedding = model.get_image_features(preprocess)
            embeddings.append(embedding)

    num_images = len(images)
    similarity_matrix = torch.zeros((num_images, num_images))

    with torch.no_grad():
        for i in range(num_images):
            for j in range(num_images):
                similarity_matrix[i, j] = torch.nn.functional.cosine_similarity(embeddings[i], embeddings[j])

    similarity_matrix = similarity_matrix.detach().numpy()
    image_labels = [f"Image {i+1}" for i in range(num_images)]
    similarity_df = pd.DataFrame(similarity_matrix, index=image_labels, columns=image_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_df, annot=True, cmap='Greens', linewidths=.5, cbar_kws={'label': 'Similarity'})
    plt.title("Image Similarity Heatmap")
    plt.show()

    
compute_similarity(images)