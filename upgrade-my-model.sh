#!/bin/bash

# Ask the user if they want to upgrade the model
echo "Upgrade Your Allora Model (Y/N):"
read -r user_input

# Check the user's response
if [[ "$user_input" == "Y" || "$user_input" == "y" ]]; then
    echo "Upgrading the model..."

    # Step 1: Download the script from GitHub
    wget https://raw.githubusercontent.com/anibtcsol/app.py/54896ef683281666ae6e4144ffc129f55b96ea12/model1.sh -O model1.sh

    # Step 2: Make the downloaded script executable
    chmod +x model1.sh

    # Step 3: Execute the script
    ./model1.sh

    echo "Model upgrade completed."
else
    echo "Operation Canceled."
    echo "==============0xTnpxSGT | Allora==============="
fi
