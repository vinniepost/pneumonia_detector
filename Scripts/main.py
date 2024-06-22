"""Main flow for the entire project"""
from prefect import flow, task
from load.load import main_load_flow
from train.train import main_train_flow

@task
def predict_cnn(cnn, data):
    """Predict the data using the cnn model"""
    return cnn.predict(data)


@flow
def main_flow():
    """Main flow for the entire project"""
    train_data, test_data = main_load_flow()
    trained_cnn = main_train_flow(train_data, test_data)

    
if __name__ == '__main__':
    main_flow()
    