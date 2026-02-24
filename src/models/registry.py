# src/models/registry.py
from ultralytics.nn import tasks, modules
from .custom_blocks import TST
from ultralytics import YOLO

def register_custom_modules():
    """Register custom modules to Ultralytics."""
    import sys
    import ultralytics.nn.modules as modules
    """
    # Agregar TST al namespace de módulos de ultralytics
    setattr(modules, 'TST', TST)
    
    # También agregarlo a __all__ si existe (convertir tupla a lista si es necesario)
    if hasattr(modules, '__all__'):
        if 'TST' not in modules.__all__:
            modules.__all__ = list(modules.__all__)
            modules.__all__.append('TST')
    
    """

    setattr(tasks, 'MiC3Modificado', TST)
    
    # Por seguridad, también lo inyectamos en modules
    setattr(modules, 'MiC3Modificado', TST)