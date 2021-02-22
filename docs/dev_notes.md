# Developer Notes

These are some notes for developers working on the colearn code repo

## Google Cloud Storage
To have access to the google cloud storage you need to setup your google authentication and
have the $GOOGLE_APPLICATION_CREDENTIALS set up correctly. 
For more details ask or see the contract-learn documentation

## Build image

To build ML server image and push to google cloud use the following command:
```
cd docker
python3 ./build.py --publish --allow_dirty
# Check this worked correctly
docker images
```

