from flask import Flask, render_template, request
from flask_mysqldb import MySQL
from sqlalchemy import create_engine
import os
from app.models.ModelDatabase import ModelDatabase

# Crear database
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_DB = os.getenv('MYSQL_DB')
engine = create_engine(f'mysql+mysqldb://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}?charset=utf8mb4')
ModelDatabase.create_database(engine)

# crear objeto flask
database = MySQL()

