"""Main flow for the entire project"""
from prefect import flow, task
from load.load import main_load_flow
from train.train import main_train_flow

@task(name='Predict CNN', log_prints=True)
def predict_cnn(cnn, data):
    """Predict the data using the cnn model"""
    return cnn.predict(data)


@flow(name='Main Flow', log_prints=True)
def main_flow():
    """Main flow for the entire project"""
    print("Starting the main flow")
    print("Starting main load flow")
    (train_data, test_data) = main_load_flow() # train_dir:str='data/train', test_dir:str='data/test', rescale:float=1./255, shear_range:float=0.2, zoom_range:float=0.2, horizontal_flip:bool=True, target_size:tuple=(1024, 1024), batch_size:int=64, class_mode:str='binary'
    print("Finished main load flow")
    print("Starting main train flow")
    trained_cnn = main_train_flow(train_data, test_data) # train_data, test_data, input_shape=[1024, 1024, 3], epochs=10
    print("Finished main train flow")
if __name__ == '__main__':
    main_flow()
