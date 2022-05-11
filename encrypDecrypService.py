from cryptography.fernet import Fernet
import configparser


class EncryptDecryptService():

    @staticmethod
    def encrypt(strVar):
        
        config = configparser.ConfigParser()

        # read the configuration file
        config.read('config.ini')
        crypt_key = config.get('OAuth1', 'crypt_key')
        fernetEncrypter = Fernet(crypt_key)

        encryp = fernetEncrypter.encrypt(str.encode(strVar))
        return encryp.decode()





    @staticmethod
    def decrypt(strVar):
        
        config = configparser.ConfigParser()
        # read the configuration file
        config.read('config.ini')
        crypt_key = config.get('OAuth1', 'crypt_key')
        fernetEncrypter = Fernet(crypt_key)

        decryp = fernetEncrypter.decrypt(str.encode(strVar))
        return decryp.decode()

