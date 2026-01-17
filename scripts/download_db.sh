#!/bin/bash

cd "$(dirname "$0")/.."

HF_URL="https://huggingface.co/datasets/effes3/chromadb/resolve/main/chromadb.zip"
TARGET_DIR="data"
ZIP_NAME="chromadb.zip"

echo "-------------------------------------------------------"
echo "📥 Downloading pre-built DB from Hugging Face..."
echo "-------------------------------------------------------"

mkdir -p "$TARGET_DIR"

curl -L "$HF_URL" -o "$ZIP_NAME"

if [ ! -s "$ZIP_NAME" ]; then
    echo "❌ Error: Download failed. Check your Hugging Face link!"
    exit 1
fi

echo "📦 Unpacking to $TARGET_DIR/..."

unzip -o "$ZIP_NAME" -d "$TARGET_DIR"

rm "$ZIP_NAME"

echo "-------------------------------------------------------"
echo "✅ Success! Database installed in: $(pwd)/data/chromadb"
echo "-------------------------------------------------------"
