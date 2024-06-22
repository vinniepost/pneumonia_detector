"""Script for loading in the images """
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from prefect import task,flow


@task
def create_datagen(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True):
    """Create the image data generator for the training and testing data"""
    train_datagen = ImageDataGenerator(rescale=rescale, shear_range=shear_range, zoom_range=zoom_range, horizontal_flip=horizontal_flip)
    test_datagen = ImageDataGenerator(rescale=rescale)
    return train_datagen, test_datagen

@task
def load_images(train_dir, test_dir, train_datagen, test_datagen, target_size=(1024, 1024), batch_size=32, class_mode='binary'):
    """Load in the images from the directories"""
    train_data = train_datagen.flow_from_directory(train_dir, target_size=target_size, batch_size=batch_size, class_mode=class_mode)
    test_data = test_datagen.flow_from_directory(test_dir, target_size=target_size, batch_size=batch_size, class_mode=class_mode)
    return train_data, test_data

@flow
def main_load_flow(train_dir:str='data/train', test_dir:str='data/test', rescale:float=1./255, shear_range:float=0.2, zoom_range:float=0.2, horizontal_flip:bool=True, target_size:tuple=(1024, 1024), batch_size:int=64, class_mode:str='binary'):
    """Main flow for loading in the images"""
    train_datagen, test_datagen = create_datagen(rescale=rescale, shear_range=shear_range, zoom_range=zoom_range, horizontal_flip=horizontal_flip)
    train_data, test_data = load_images(train_dir, test_dir, train_datagen, test_datagen, target_size=target_size, batch_size=batch_size, class_mode=class_mode)
    return train_data, test_data

if __name__ == '__main__':
    main_load_flow()
    