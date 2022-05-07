#!/bin/bash

git checkout devmka
git add .
echo "INFO:Enter the commit message."
read msg
git commit -m"${msg}"
echo "INFO:Pushing data to the remote origin of devmka."
git push origin devmka
