from torchsummary import summary
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import os
import torch.nn as nn
import torch.nn.functional as F

INSTRUMENTS = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'vocal']

# Creamos una clase de clasificacion de imagenes para que el modelo pueda hacer uso de esta
class ImageClassificationBase(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.to(device)

    # Definimos el training step
    def training_step(self, batch):
        images, labels = batch[0].to(self.device), batch[1].to(self.device)
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculamos la perdida
        return loss

    # Definimos la validacion del step (definir la entropía para la perdida) para ajustar mejor la accuracy
    def validation_step(self, batch):
        images, labels = batch[0].to(self.device), batch[1].to(self.device)
        out = self(images)  # Generamos las predicciones
        loss = F.cross_entropy(out, labels)  # Calculamos la perdida
        acc = accuracy(out, labels)  # Calculamos la accuracy
        #         self.images=images
        #         self.labels=labels
        return {'val_loss': loss.detach(), 'val_acc': acc}

    # Validacion de la epoca
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Añadimos todas las perdidas de la epoca
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Añadimos todas las accuracy s de la epoca
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    # Cada final de epoca mostrar los detalles
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


class ModeloCNN(ImageClassificationBase):
    # Creamos el modelo, con cuantas capas queramos
    #     images=None
    #     labels=None
    def __init__(self, device):
        super().__init__(device)
        self.network = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(82944, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.LogSoftmax(dim=1)
        )

    # Devolvemos la topología del modelo
    def forward(self, xb):
        return self.network(xb)

# Definimos nuestra propia funcion para evaluar la accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor((torch.sum(preds == labels).item() / len(preds))) # Contamos la cantidad de predicciones que ha acertado y lo dividimos entre el numero de predicciones hechas

# Evaluamos el modelo para 1 batch
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval() # Método de pytorch
    outputs = [model.validation_step(batch) for batch in val_loader] # Aquí tendremos los outputs de cada trainig step
    #     print(outputs)
    return model.validation_epoch_end(outputs) # Aquí tendremos los outputs del final de la epoca

# Definimos el fit para nuestros datos al modelo
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = [] # Creamos una lista para llevar una "memoria" de cada epoca
    optimizer = opt_func(model.parameters(), lr) # Definimos el optimizador
    for epoch in range(epochs):

        model.train() # Método de pytorch para entrenar el modelo cada época
        train_losses = [] # Lista para poder ir añadiendo las pérdidas al modelo

        for batch in train_loader: # Evaluamos todas las imagenes del dataset de entrenamiento
            loss = model.training_step(batch) # Entrenamos un paso y le pasamos un batch de imagenes
            train_losses.append(loss) # Añadimos las pérdidas
            loss.backward() # Método de pytorch donde pretende analizar dloss/dx para cada parametro x
            optimizer.step() # Método de pytorch para actualizar los parametros
            optimizer.zero_grad() # Método de pytorch que setea los gradientes de todos los tensores optimizados a 0

        # Evaluamos los datos y añadimos las pérdidas de la epoca
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        
        # Recopilamos los datos de la epoca y los añadimos al historial
        model.epoch_end(epoch, result)
        history.append(result)

    return history

# Hacemos un predict sobre un input y devolvemos el índice de clase predicho
def single_predict(model, input):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
    return predicted_index.item()


# Tratamos de predecir el audio en espectrogramas, donde tiene las imagenes en <data_dir/espectrogramas>
# De nuevo, ajustamos el device al que tengamos disponible
def predict(model, data_dir, device):
    dataset = ImageFolder(data_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
    ]))
    data=DataLoader(dataset, 1, num_workers = 4, pin_memory = True)

    predicciones = []
    for batch in data:
        indicePredicho=single_predict(model,batch[0].to(device))
        predicciones.append(indicePredicho)
    
    return predicciones

# Tratamos de devolver el índice que más veces ha predicho, ya que habrá veces que prediga mal
def most_common(predicciones):
    return INSTRUMENTS[max(set(predicciones), key=predicciones.count)]