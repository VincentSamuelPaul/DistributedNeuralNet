package main

import (
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server struct {
		Port               int    `yaml:"port"`
		Workers            int    `yaml:"workers"`
		CheckpointInterval int    `yaml:"checkpoint_interval"`
		CheckpointPath     string `yaml:"checkpoint_path"`
	} `yaml:"server"`
}

func loadConfig() Config {
	path := os.Getenv("CONFIG_PATH")
	if path == "" {
		path = "/app/config.yaml"
	}

	data, err := os.ReadFile(path)
	if err != nil {
		panic("failed to read config: " + err.Error())
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		panic("failed to parse config: " + err.Error())
	}

	return cfg
}
