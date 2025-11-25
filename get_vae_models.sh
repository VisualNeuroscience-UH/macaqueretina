#!/bin/bash

# Prompt user for confirmation
echo "You are about to download about 800MB of data to your ./macaqueretina/retina/vae_statistics directory."
read -p "Continue? [y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 0
fi

# Configuration
ZIP_URL="https://datacloud.helsinki.fi/index.php/s/NrYCcotR4s7bFHt/download"
ZIP_FILE="vae_statistics.zip"
TARGET_DIR="./macaqueretina/retina/"
EXPECTED_SHA256="a658f60066719c2f55a95ca9792186fc0cc213a1e80b0d3344542f6d04b752b5"  

# Step 1: Download the ZIP file
echo "Downloading $ZIP_URL..."
wget "$ZIP_URL" -O "$ZIP_FILE" || { echo "Download failed"; exit 1; }

# Step 2: Verify the SHA-256 checksum
echo "Verifying checksum..."
DOWNLOADED_SHA256=$(sha256sum "$ZIP_FILE" | awk '{ print $1 }')

if [ "$DOWNLOADED_SHA256" != "$EXPECTED_SHA256" ]; then
    echo "Checksum verification failed!"
    echo "Expected: $EXPECTED_SHA256"
    echo "Got:      $DOWNLOADED_SHA256"
    rm -f "$ZIP_FILE"
    exit 1
fi

# Step 3: Extract the ZIP file
echo "Extracting $ZIP_FILE to $TARGET_DIR..."
unzip "$ZIP_FILE" -d "$TARGET_DIR" || { echo "Extraction failed -- you need to run this script from the root of the repository"; exit 1; }

# Step 4: Remove the ZIP file
echo "Removing $ZIP_FILE..."
rm -f "$ZIP_FILE"

echo "Done!"
