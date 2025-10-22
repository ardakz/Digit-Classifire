# scr/train.py
import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from scr.model import build_lenet
from scr.data import load_and_preprocess
import matplotlib.pyplot as plt
from keras.models import load_model



FORCE_RETRAIN = False
MODEL_PATH = 'model.0-9digit.keras'

# lr_schedule
def lr_schedule(epoch):
    if epoch <= 2:
        lr = 5e-4
    elif epoch > 2 and epoch <= 5:
        lr = 2e-4
    elif epoch > 5 and epoch <= 9:
        lr = 5e-5
    else:
        lr = 1e-5
    return lr


def main():
    X_train,y_train,X_test,y_test = load_and_preprocess()

    # Если модель уже сохранена и мы не хотим принудительно тренить — загрузим и оценим
    if (not FORCE_RETRAIN) and os.path.exists(MODEL_PATH):
        print(f"Found saved model at '{MODEL_PATH}'. Loading and evaluating (no retrain).")
        model = load_model(MODEL_PATH)
        loss, acc = model.evaluate(X_test, y_test, verbose=2)
        print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")
        return  # выходим — не тренируем заново
    

    input_shape = X_train.shape[1:]
    model = build_lenet(input_shape=input_shape,num_classes=10)

    lr_scheduler = LearningRateScheduler(lr_schedule)
    checkpointer = ModelCheckpoint(filepath='model.0-9digit.keras', verbose=1,
                               save_best_only=True)

    hist = model.fit(X_train, y_train, batch_size=32, epochs=20,
            validation_data=(X_test, y_test), callbacks=[checkpointer, lr_scheduler],
            verbose=2, shuffle=True)
    
    #Simple plots
    plt.plot(hist.history['accuracy'],label='train_acc')
    plt.plot(hist.history['val_accuracy'], label='val_acc')
    plt.legend();plt.show()


if __name__ == "__main__":
    main()




