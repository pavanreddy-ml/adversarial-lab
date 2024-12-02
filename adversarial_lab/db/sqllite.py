import sqlite3
import json
import os

class SqlliteDB:
    db_types = {
        'int': 'INTEGER',
        'str': 'TEXT',
        'float': 'REAL',
        'bool': 'INTEGER',
        'json': 'TEXT',
        'blob': 'BLOB'
    }

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            print(f"Database {self.db_path} does not exist. Creating new database.")
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()

    def validate_connection(self) -> bool:
        return True

    def create_table(self, 
                     table_name: str, 
                     schema: dict, 
                     force: bool = False
                     ) -> None:
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,)
        )
        table_exists = self.cursor.fetchone()

        if table_exists:
            if force:
                print(f"Warning: Table '{table_name}' already exists. Recreating the table.")
                self.delete_table(table_name)
            else:
                raise ValueError(f"Table '{table_name}' already exists. Use force=True to delete and recreate it.")

        columns = []
        for column_name, column_type in schema.items():
            if column_type not in self.db_types:
                raise ValueError(f"Unsupported column type: {column_type}")
            columns.append(f"{column_name} {self.db_types[column_type]}")
        columns_str = ", ".join(columns)
        query = f"CREATE TABLE {table_name} ({columns_str});"
        self.cursor.execute(query)
        self.connection.commit()

    def delete_table(self, 
                     table_name: str
                     )-> None:
        query = f"DROP TABLE IF EXISTS {table_name};"
        self.cursor.execute(query)
        self.connection.commit()

    def insert(self, 
               table_name: str, 
               data: dict
               ) -> None:
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        values = []
        for value in data.values():
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            values.append(value)
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders});"
        self.cursor.execute(query, values)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()
