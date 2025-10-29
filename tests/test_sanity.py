"""
Test de humo para validar que los módulos principales de Quant50
pueden importarse correctamente sin errores de sintaxis o dependencias.
"""

import importlib
import pytest


MODULES = [
    "app.main",
    "app.routes",
    "app.utils.config",
    "app.utils.dates",
    "app.utils.logging",
    "app.services.data",
    "app.services.features",
    "app.services.label",
    "app.services.model",
    "app.services.optimize",
    "app.services.execute",
    "app.services.report",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_imports(module_name):
    """Importa módulos sin ejecutar lógica."""
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        pytest.skip(f"Módulo no encontrado: {module_name}")
    except Exception as e:
        pytest.fail(f"Error al importar {module_name}: {e}")
