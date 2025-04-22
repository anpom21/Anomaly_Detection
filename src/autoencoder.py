



from Preprocess import Preprocess

if __name__ == "__main__":
    process = Preprocess(path='Datasets\Dataset003\Train', n_lights=24, width=224, height=224, top_light=True)
    images = process.process_images('Datasets\Dataset003\Train24Lights', n_images=24)
    print('Images shape:', images.shape)