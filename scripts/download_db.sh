#!/bin/bash

cd "$(dirname "$0")/.."

HF_URL="https://huggingface.co/datasets/effes3/chromadb/resolve/main/chromadb.zip"
TARGET_DIR="data/chromadb/"
ZIP_NAME="chromadb.zip"

echo "-------------------------------------------------------"
echo "📥 Downloading pre-built DB from Hugging Face..."
echo "-------------------------------------------------------"

rm -rf "$TARGET_DIR"

mkdir -p "$TARGET_DIR"

cd "$TARGET_DIR"

curl -L "$HF_URL" -o "$ZIP_NAME"

if [ ! -s "$ZIP_NAME" ]; then
    echo "❌ Error: Download failed. Check your Hugging Face link!"
    exit 1
fi

echo "📦 Unpacking in current directory..."
unzip -o "$ZIP_NAME"
rm "$ZIP_NAME"

# Возвращаемся назад
cd - > /dev/null

echo "-------------------------------------------------------"
echo "✅ Success! Database installed in: $(pwd)/$TARGET_DIR"
echo "-------------------------------------------------------"
