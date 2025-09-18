package main

import (
	"log"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// --------------------
// Rate limiting setup
// --------------------
type clientInfo struct {
	count   int
	resetAt time.Time
}

var clients = make(map[string]*clientInfo)
var mu sync.Mutex

// --------------------
// Middleware wrapper
// --------------------
func SecurityMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Enforce max request body size centrally
		switch r.Method {
		case http.MethodPost, http.MethodPut, http.MethodPatch:
			r.Body = http.MaxBytesReader(w, r.Body, 10<<20) // 10MB
		}

		if !CheckAuth(w, r) {
			return
		}

		if !RateLimit(w, r) {
			return
		}

		if !ValidateRequest(w, r) {
			return
		}

		AuditLog(r)

		next.ServeHTTP(w, r)
	})
}

// --------------------
// Individual checks
// --------------------

// Auth: simple API key
func CheckAuth(w http.ResponseWriter, r *http.Request) bool {
	apiKey := r.Header.Get("Authorization")
	expectedKey := os.Getenv("API_KEY") // load from .env
	if expectedKey == "" {
		// If no API key configured, allow (development convenience) but log
		log.Printf("[WARN] API_KEY not configured; allowing request")
		return true
	}
	if apiKey != "Bearer "+expectedKey {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		log.Printf("[AUTH] unauthorized from %s path=%s", r.RemoteAddr, r.URL.Path)
		return false
	}
	return true
}

func clientIPFromRequest(r *http.Request) string {
	xff := r.Header.Get("X-Forwarded-For")
	if xff != "" {
		parts := strings.Split(xff, ",")
		return strings.TrimSpace(parts[0])
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

// Rate limiting per client (5 requests/minute)
func RateLimit(w http.ResponseWriter, r *http.Request) bool {
	ip := clientIPFromRequest(r)
	mu.Lock()
	defer mu.Unlock()

	ci, ok := clients[ip]
	now := time.Now()
	if !ok || now.After(ci.resetAt) {
		clients[ip] = &clientInfo{count: 1, resetAt: now.Add(1 * time.Minute)}
		return true
	}
	if ci.count >= 5 {
		http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
		log.Printf("[RATE] limit exceeded ip=%s path=%s", ip, r.URL.Path)
		return false
	}
	ci.count++
	return true
}

// Validate request size and method constraints
func ValidateRequest(w http.ResponseWriter, r *http.Request) bool {
	// Additional light validation; body size already enforced by middleware
	if r.ContentLength > 0 && r.ContentLength > 10<<20 { // 10MB
		http.Error(w, "Request too large", http.StatusRequestEntityTooLarge)
		return false
	}
	return true
}

// Audit logging
func AuditLog(r *http.Request) {
	log.Printf("[%s] %s %s from %s",
		time.Now().Format(time.RFC3339),
		r.Method,
		r.URL.Path,
		r.RemoteAddr,
	)
}

