"use client";

import { useState, useEffect, useCallback } from "react";
import Image from "next/image";
import { albums } from "@/data/photos.generated";

export default function PhotosPage() {
  const [selectedAlbum, setSelectedAlbumState] = useState<number | null>(null);
  const [selectedPhoto, setSelectedPhotoState] = useState<string | null>(null);

  const setSelectedAlbum = useCallback((idx: number | null) => {
    setSelectedAlbumState(idx);
    setSelectedPhotoState(null);
    if (idx !== null) {
      window.history.pushState({ album: idx }, "");
    } else {
      window.history.pushState({ album: null }, "");
    }
  }, []);

  const setSelectedPhoto = useCallback((src: string | null) => {
    setSelectedPhotoState(src);
    if (src) {
      window.history.pushState({ album: selectedAlbum, photo: src }, "");
    }
  }, [selectedAlbum]);

  useEffect(() => {
    const onPopState = (e: PopStateEvent) => {
      const state = e.state;
      if (state && state.album !== undefined) {
        setSelectedAlbumState(state.album);
        setSelectedPhotoState(state.photo ?? null);
      } else {
        setSelectedAlbumState(null);
        setSelectedPhotoState(null);
      }
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-4 py-16">
      <h1 className="text-3xl md:text-4xl font-bold mb-2" style={{ color: "var(--text-primary)" }}>
        Photos
      </h1>
      <p className="text-lg mb-12" style={{ color: "var(--text-secondary)" }}>연구실 사진</p>

      {selectedAlbum === null ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {albums.map((album, i) => (
            <button
              key={i}
              onClick={() => setSelectedAlbum(i)}
              className="rounded-xl border overflow-hidden transition-shadow hover:shadow-lg text-left"
              style={{ backgroundColor: "var(--bg-card)", borderColor: "var(--border)" }}
            >
              <div className="aspect-video relative overflow-hidden">
                {album.photos.length >= 4 ? (
                  <div className="grid grid-cols-2 grid-rows-2 absolute inset-0">
                    {[0, 1, 2, 3].map((idx) => (
                      <div key={idx} className="relative">
                        <Image
                          src={`/images/photos/${album.folder}/${album.photos[idx]}`}
                          alt={album.title}
                          fill
                          className="object-cover"
                        />
                      </div>
                    ))}
                  </div>
                ) : album.photos.length >= 2 ? (
                  <div className="grid grid-cols-2 absolute inset-0">
                    {[0, 1].map((idx) => (
                      <div key={idx} className="relative">
                        <Image
                          src={`/images/photos/${album.folder}/${album.photos[idx]}`}
                          alt={album.title}
                          fill
                          className="object-cover"
                        />
                      </div>
                    ))}
                  </div>
                ) : (
                  <Image
                    src={`/images/photos/${album.folder}/${album.photos[0]}`}
                    alt={album.title}
                    fill
                    className="object-cover"
                  />
                )}
              </div>
              <div className="p-4">
                <h3 className="font-semibold" style={{ color: "var(--text-primary)" }}>{album.title}</h3>
                <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>
                  {album.date} &middot; {album.photos.length} photos
                </p>
              </div>
            </button>
          ))}
        </div>
      ) : (
        <div>
          <button
            onClick={() => window.history.back()}
            className="mb-6 text-sm font-medium flex items-center gap-1"
            style={{ color: "var(--accent)" }}
          >
            &larr; All Albums
          </button>
          <h2 className="text-2xl font-bold mb-1" style={{ color: "var(--text-primary)" }}>
            {albums[selectedAlbum].title}
          </h2>
          <p className="text-sm mb-6" style={{ color: "var(--text-muted)" }}>
            {albums[selectedAlbum].date} &middot; {albums[selectedAlbum].photos.length} photos
          </p>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            {albums[selectedAlbum].photos.map((photo, j) => (
              <button
                key={j}
                onClick={() => setSelectedPhoto(`/images/photos/${albums[selectedAlbum!].folder}/${photo}`)}
                className="aspect-square relative rounded-lg overflow-hidden hover:opacity-90 transition-opacity"
              >
                <Image
                  src={`/images/photos/${albums[selectedAlbum].folder}/${photo}`}
                  alt={`Photo ${j + 1}`}
                  fill
                  className="object-cover"
                />
              </button>
            ))}
          </div>
        </div>
      )}

      {selectedPhoto && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ backgroundColor: "rgba(0,0,0,0.85)" }}
          onClick={() => window.history.back()}
        >
          <button
            className="absolute top-4 right-4 text-white text-3xl font-light hover:opacity-80"
            onClick={() => window.history.back()}
          >
            &times;
          </button>
          <div className="relative max-w-5xl max-h-[90vh] w-full h-full">
            <Image
              src={selectedPhoto}
              alt="Full size photo"
              fill
              className="object-contain"
            />
          </div>
        </div>
      )}
    </div>
  );
}
