from model.model import UpscaleGAN
import glob


def epoch_callback(gan_model, epoch, d_loss, g_loss):
    print ("%d: %f, %f" % (epoch, d_loss, g_loss))
    gan_model.test('./GOPRO_Large/test/GOPR0868_11_00/sharp/000001.png', './result-%d.png' % epoch, 2)
    gan_model.save(epoch, d_loss, g_loss, './')



gan = UpscaleGAN()
image_filepaths = glob.glob('./GOPRO_Large/train/*/sharp/*.png')

gan.train(image_filepaths, batch_size=8, training_ratio=2, epochs=3000, epoch_callback=epoch_callback)
