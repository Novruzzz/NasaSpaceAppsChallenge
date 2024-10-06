package main

import (
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"log"
	"net/http"
	"strconv"
)

func main() {
	r := chi.NewRouter()
	r.Use(middleware.Logger)

	r.Put("/blob", putBlobHandler)
	r.Get("/blob/{key}", getBlobHandler)
	r.Delete("/blob/{key}", deleteBlobHandler)

	port := 8080
	log.Println("Starting server on port", port)
	err := http.ListenAndServe(":"+strconv.Itoa(port), r)
	if err != nil {
		return
	}
}

func putBlobHandler(w http.ResponseWriter, r *http.Request) {
	file, _, err := r.FormFile("file")
	key, err := Put(file)
	if key == "" {
		http.Error(w, "Internal error", http.StatusInternalServerError)
		return
	}
	_, err = w.Write([]byte(key))
	if err != nil {
		return
	}
}

func getBlobHandler(w http.ResponseWriter, r *http.Request) {
	key := chi.URLParam(r, "key")
	data, err := Get(key)
	if err != nil {
		http.Error(w, "Blob not found", http.StatusNotFound)
		return
	}
	_, err = w.Write(data)
	if err != nil {
		return
	}
}

func deleteBlobHandler(w http.ResponseWriter, r *http.Request) {
	key := chi.URLParam(r, "key")
	err := Delete(key)
	if err != nil {
		http.Error(w, "Blob not found", http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
