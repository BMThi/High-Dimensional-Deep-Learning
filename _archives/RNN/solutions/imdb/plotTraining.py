def plotTraining(history):
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], 'b', linestyle="--", label='Training accuracy')
    plt.plot(history.history['val_accuracy'], 'g', label='Validation accuracy')
    plt.title('Model accuracy') 
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], 'b', linestyle="--", label='Training loss')
    plt.plot(history.history['val_loss'], 'g', label='Validation loss')
    plt.title('Model loss') 
    plt.xlabel('epochs')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()