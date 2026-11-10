"""Le contrôle doit porter aussi sur les données intégrées au code."""
from scripts.check_publication import allowed, sensitive_content


def test_rejects_data_documents_models_and_hidden_files():
    for name in ('data/table.csv', 'src/table.csv', 'note.docx', 'outputs/model.pt',
                 'src/.env', 'src/cache.bin'):
        assert not allowed(name)
    assert allowed('src/data/dhs/01_prepare_dhs.py')


def test_detects_embedded_coordinates_with_whitespace():
    assert sensitive_content('ee.Geometry.Point(' + ' [ -4.123456, 5.234567 ] )')
    assert sensitive_content('wealth_mean:' + ' -0.123')
    assert not sensitive_content('ee.Geometry.Point([row.longitude, row.latitude])')
