DROP TABLE IF EXISTS "simulations";
CREATE TABLE "simulations"
(
    time            SMALLINT NOT NULL,
    bullet_id       INTEGER  NOT NULL,
    weapon_id       INTEGER  NOT NULL,
    location_x      DOUBLE PRECISION,
    location_y      DOUBLE PRECISION,
    damage          SMALLINT,
    distance        DOUBLE PRECISION,
    velocity        DOUBLE PRECISION,
    energy_transfer DOUBLE PRECISION,
    power_left      DOUBLE PRECISION,

    CONSTRAINT fk_bullet
        FOREIGN KEY (bullet_id)
            REFERENCES bullet (id),

    CONSTRAINT fk_weapon
        FOREIGN KEY (weapon_id)
            REFERENCES weapon (id)
);

CREATE INDEX ON "simulations" (time DESC);
CREATE INDEX ON "simulations" (weapon_id, time DESC);
CREATE INDEX ON "simulations" (bullet_id, time DESC);
SELECT create_hypertable(
               'simulations', 'time',
               chunk_time_interval => 32767);
