CREATE TABLE if not exists predicted_prices (id SERIAL PRIMARY KEY, 
                              predicted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, 
                              city varchar(40) NOT NULL,
                              prices JSON NOT NULL);