#!/bin/bash
gunicorn app.main:app --workers 1 --bind 0.0.0.0:8000
