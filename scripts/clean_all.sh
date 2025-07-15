#!/bin/bash

# Clean all temporary/cache/log directories for Pinocchio

echo "Cleaning ./memories ..."
rm -rf ./memories/*

echo "Cleaning ./prompts ..."
rm -rf ./prompts/*

echo "Cleaning ./sessions ..."
rm -rf ./sessions/*

echo "Cleaning ./logs ..."
rm -rf ./logs/*

echo "All temporary storage cleaned."
