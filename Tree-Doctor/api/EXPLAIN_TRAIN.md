# Understanding `train.py` with Code Snippets

This file is the **Teacher**. It trains the AI model using the images in your `dataset` folder.

---

### 1. Loading the Textbooks (Dataset)
We load the images from the `dataset` folder. We tell the AI to look at them as 256x256 pixel squares.

```python
dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    shuffle=True,
    image_size=(256, 256),
    batch_size=32
)
```

---

### 2. Splitting the Work
We split the images into three groups:
1.  **Training (80%)**: To learn from.
2.  **Validation (10%)**: To test itself while learning.
3.  **Testing (10%)**: For the final exam.

```python
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, ...):
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds
```

---

### 3. Data Augmentation (Tricks)
We randomly flip and flip the images so the model learns that a leaf is a leaf, no matter how it's oriented.

```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])
```

---

### 4. Building the Brain (Model Architecture)
We build the model layer by layer. `Conv2D` layers are the "eyes" that detect patterns (lines, spots). `MaxPooling2D` layers summarize the information.

```python
model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape[1:]),
    layers.MaxPooling2D((2,2)),
    # ... more layers ...
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax') # Final output layer
])
```

---

### 5. The Training Loop
This is where the learning happens. We run the training for a number of `EPOCHS` (rounds).

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=10,
    batch_size=32,
    verbose=1,
    validation_data=val_ds
)
```

---

### 6. Saving the Graduate
Once training is done, we save the model to a file so `main.py` can use it later.

```python
model.save("models/potato_model.keras")
```
