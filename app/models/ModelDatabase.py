

class ModelDatabase():
    
    @classmethod
    def create_database(self, engine):
    
        for table_name in ['usuarios','tweets','models']:
            query = f"SHOW TABLES LIKE '{table_name}';"
            conn = engine.connect()
            try:
                cursor = conn.execute(query).cursor
                row = [r for r in cursor]
            except Exception as e:
                raise(e)
            else:
                if row == [] and table_name == 'usuarios':
                    query = """
                    CREATE TABLE `usuarios` (
                    `user_id` bigint(20) NOT NULL,
                    `fullname` varchar(50) CHARACTER SET utf8 COLLATE utf8_unicode_ci DEFAULT NULL,
                    `username` varchar(20) CHARACTER SET utf8 COLLATE utf8_unicode_ci NOT NULL,
                    `password` varchar(102) CHARACTER SET utf8 COLLATE utf8_unicode_ci NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                    conn.execute(query)
                    
                elif row == [] and table_name == 'tweets':
                    query = """
                    CREATE TABLE `tweets` (
                        `id` bigint(20) NOT NULL COMMENT 'Identificador del tweet (proporcionado por la API).',
                        `topic` varchar(50) CHARACTER SET utf8 COLLATE utf8_unicode_ci NOT NULL COMMENT 'Búsqueda del usuario',
                        `datetime` datetime NOT NULL COMMENT 'Fecha de publicación del tweet',
                        `user` varchar(20) CHARACTER SET utf8 COLLATE utf8_unicode_ci NOT NULL COMMENT 'Nombre del usuario que ha publicado el tweet',
                        `text` varchar(500) CHARACTER SET utf8 COLLATE utf8_unicode_ci NOT NULL COMMENT 'Texto del tweet',
                        `model_id` smallint(6) NOT NULL COMMENT 'Identificador del modelo',
                        `prediction` float(10,0) NOT NULL COMMENT 'Sentimiento del tweet. Positivo = 1, Negativo = 0',
                        `user_id` bigint(11) NOT NULL
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                    conn.execute(query)

                    query = """
                    ALTER TABLE `usuarios` ADD PRIMARY KEY (`user_id`);
                    """

                    query = """
                    ALTER TABLE `usuarios`
                    MODIFY `user_id` bigint NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;
                    COMMIT;
                    """


                    
                elif row == [] and table_name == 'models':
                    query = """
                    CREATE TABLE `models` (
                    `id` int(11) NOT NULL COMMENT 'Identificador del modelo',
                    `name` varchar(50) NOT NULL COMMENT 'Nombre del modelo'
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                    conn.execute(query)
                    
                    query = """
                    INSERT INTO `models` (`id`, `name`) VALUES
                    (1, 'gnews_swivel_20dim'),
                    (2, 'gnews_swivel_20dim_with_oov'),
                    (3, 'universal_sentence_encoder');
                    """
                    conn.execute(query)

                    query = """
                    ALTER TABLE `models` ADD PRIMARY KEY (`id`);
                    """
                    conn.execute(query)

                    query = """
                    ALTER TABLE `models`
                    MODIFY `id` int(11) NOT NULL AUTO_INCREMENT COMMENT 'Identificador del modelo', AUTO_INCREMENT=4;
                    """
                    conn.execute(query)


                else:
                    continue
        