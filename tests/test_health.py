from django.test import Client


def test_health_check():
    client = Client()
    response = client.get("/zap/health/")
    assert response.status_code == 200
    assert response.content == b"OK"
