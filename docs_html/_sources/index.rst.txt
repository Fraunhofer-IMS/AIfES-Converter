Welcome to AIfES-Converters documentation!
===========================================
This is a convert tool to create `AIfES`_ (**A**\ rtificial **I**\ ntelligence **f**\ or **E**\ mbedded **S**\ ystems) models for direct use in the Arduino IDE or other IDEs.
It can read Feed Forward Neural Network (FFNN) models from Keras and PyTorch and converts them to AIfES models, which are exported in header files. Those header
files can then be added to your Project in any IDE and can be used there.

.. _AIfES: https://aifes.ai

Check out the :doc:`quickstart` section for a summary.
A detailed description is available in the :doc:`usage` section. Advanced function of this convert, like including the
CMSIS for ARM-based controller or to quantize your model to Q7 are explained in the :doc:`advanced_usage` section.
All possible converter functions are documented in the :doc:`functions` section.



.. toctree::
   :hidden:
   :maxdepth: 3
   
   
   quickstart
   usage
   advanced_usage
   functions
