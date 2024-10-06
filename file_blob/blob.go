package main

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
)

const (
	DEFAULT_NAMESPACE = "default"
)

var space = DEFAULT_NAMESPACE

func basepath(key string) (string, error) {

	if len(key) == 0 {
		return "", errors.New("glbs basepath(): 0 length key")
	}

	return fmt.Sprintf("%s/%s", space, key[0:2]), nil

} // basepath

func path(key string) (string, error) {

	if len(key) == 0 {
		return "", errors.New("glbs path(): 0 length key")
	}

	return fmt.Sprintf("%s/%s/%s", space, key[0:2], key), nil

} // path

func hash(data []byte) string {

	digest := sha256.New()

	digest.Write(data)

	return hex.EncodeToString(digest.Sum(nil))

} // hash

func SetNamespace(namespace string) {
	space = namespace
} // SetNamespace

func Put(file io.Reader) (string, error) {
	data, err := io.ReadAll(file)
	if err != nil {
		return "", fmt.Errorf("glbs Put(): %w", err)
	}

	key := hash(data)
	if len(key) == 0 {
		return "", errors.New("glbs Put(): failed to generate key")
	}

	if Exists(key) {
		return key, nil
	}

	bp, err := basepath(key)
	if err != nil {
		return "", fmt.Errorf("glbs Put(): %w", err)
	}

	err = os.MkdirAll(bp, 0755)
	if err != nil {
		return "", fmt.Errorf("glbs Put(): %w", err)
	}

	p, err := path(key)
	if err != nil {
		return "", fmt.Errorf("glbs Put(): %w", err)
	}

	blob, err := os.Create(p)
	if err != nil {
		return "", fmt.Errorf("glbs Put(): %w", err)
	}
	defer func(blob *os.File) {
		err := blob.Close()
		if err != nil {
			return
		}
	}(blob)

	_, err = blob.Write(data)
	if err != nil {
		return "", fmt.Errorf("glbs Put(): %w", err)
	}

	return key, nil
} // Put

func Get(key string) ([]byte, error) {

	p, err := path(key)

	if err != nil {
		return nil, fmt.Errorf("glbs Get(): %w", err)
	}

	file, err := os.Open(p)
	if err != nil {
		return nil, fmt.Errorf("glbs Get(): %w", err)
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			return
		}
	}(file)

	data, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("glbs Get(): %w", err)
	}

	return data, nil

} // Get

func Delete(key string) error {
	p, err := path(key)
	if err != nil {
		return fmt.Errorf("glbs Delete(): %w", err)
	}

	err = os.Remove(p)
	if err != nil {
		return fmt.Errorf("glbs Delete(): %w", err)
	}

	return nil
} // Delete

func Exists(key string) bool {
	p, err := path(key)
	if err != nil {
		return false
	}

	_, err = os.Stat(p)
	return !os.IsNotExist(err)
} // Exists++
