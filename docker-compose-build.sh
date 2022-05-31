#!/bin/bash

#copy utils module into the single api folders
cp -r ./utils ./prod-api/
cp -r ./utils ./trainer/
cp -r ./utils ./data-api/

#copy requirements.txt into the single api folders (adapt to single api req?)
cp requirements.txt ./prod-api/
cp requirements.txt ./trainer/
cp requirements.txt ./data-api/

docker-compose build

# delete all copied files
rm -r ./data-api/utils/
rm -r ./trainer/utils/
rm -r ./prod-api/utils/

rm -r ./data-api/requirements.txt
rm -r ./trainer/requirements.txt
rm -r ./prod-api/requirements.txt