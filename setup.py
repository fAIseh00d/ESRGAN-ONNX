from setuptools import setup

setup(
    name='esrgan-onnx',
    version='0.1.0',
    author='fAIseh00d',
    author_email='faisehood@pm.me',
    description='ESRGAN implemented with ONNX',
    py_modules=['esrgan_onnx'],
    install_requires=['numpy','pillow'],
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.10',
    ],
    license='Apache 2.0',
)