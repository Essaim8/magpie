package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// A tool whose parameters declare properties or required but name no type
// is turned away by an upstream validating tools as OpenAI's and
// Anthropic's APIs do: Codex desktop's codex_app tools carry such schemas
// (zod's toJSONSchema leaves the type off), on the root or a nested field,
// and magpie's Mirasim provider answered "Mirasim: The request was rejected
// as invalid." for mirasim/gemini-4-argon through Codex, though the same
// model took the same tools from clients that carried their own (#2137).
// The type is written into every object that names none, and nothing else
// changes.
func TestToolObjectTypesWritten(t *testing.T) {
	for name, tc := range map[string]struct {
		proto provider.Protocol
		body  string
		want  []string // substrings that must be present
		keep  []string // substrings that must be unchanged
	}{
		"responses root and nested": {
			proto: provider.Responses,
			body: `{"model":"m","input":[],"tools":[{"type":"function","name":"t","parameters":` +
				`{"properties":{"a":{"type":"string"},"b":{"properties":{"c":{"type":"string"}},"required":["c"]}},"required":["a"]}}]}`,
			want: []string{
				`{"properties":{"a":{"type":"string"},"b":{"properties":{"c":{"type":"string"}},"required":["c"],"type":"object"}},"required":["a"],"type":"object"}`,
			},
		},
		"responses namespace nested tools": {
			proto: provider.Responses,
			body: `{"model":"m","input":[],"tools":[{"type":"namespace","name":"ns","tools":[` +
				`{"type":"function","name":"f","parameters":{"properties":{"a":{"type":"string"}}}}]}]}`,
			want: []string{`{"properties":{"a":{"type":"string"}},"type":"object"}`},
		},
		"chat function": {
			proto: provider.Chat,
			body: `{"model":"m","messages":[],"tools":[{"type":"function","function":{"name":"t",` +
				`"parameters":{"required":["a"],"properties":{"a":{"type":"string"}}}}}]}`,
			want: []string{`"parameters":{"properties":{"a":{"type":"string"}},"required":["a"],"type":"object"}`},
		},
		"anthropic input_schema": {
			proto: provider.Anthropic,
			body: `{"model":"m","max_tokens":8,"messages":[],"tools":[{"name":"t",` +
				`"input_schema":{"type":"object","properties":{"a":{"type":"array","items":{"properties":{"b":{"type":"string"}}}}}}}]}`,
			want: []string{`"items":{"properties":{"b":{"type":"string"}},"type":"object"}`},
		},
		"a typed schema is untouched": {
			proto: provider.Responses,
			body: `{"model":"m","input":[],"tools":[{"type":"function","name":"t","parameters":` +
				`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}}]}`,
			keep: []string{`"parameters":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}`},
		},
		"a union and an enum keep their own shape": {
			proto: provider.Responses,
			body: `{"model":"m","input":[],"tools":[{"type":"function","name":"t","parameters":` +
				`{"properties":{"a":{"anyOf":[{"type":"string"},{"type":"null"}]},"b":{"enum":["x","y"]}}}}]}`,
			want: []string{`"a":{"anyOf":[{"type":"string"},{"type":"null"}]}`, `"b":{"enum":["x","y"]}`},
		},
		"a body with no tool schema is byte for byte": {
			proto: provider.Responses,
			body:  `{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`,
			keep:  []string{`"type":"input_text"`},
		},
	} {
		t.Run(name, func(t *testing.T) {
			out, changed := objectTypesBody(tc.proto, []byte(tc.body))
			for _, w := range tc.want {
				if !strings.Contains(string(out), w) {
					t.Errorf("wanted %s in %s", w, out)
				}
			}
			for _, k := range tc.keep {
				if !strings.Contains(string(out), k) {
					t.Errorf("wanted %s kept in %s", k, out)
				}
			}
			wantChanged := len(tc.want) > 0
			if changed != wantChanged {
				t.Errorf("changed = %v, want %v (%s)", changed, wantChanged, out)
			}
			if !changed && string(out) != tc.body {
				t.Errorf("unchanged body was rewritten: %s", out)
			}
			if !json.Valid(out) {
				t.Errorf("not valid JSON: %s", out)
			}
		})
	}
}

// Codex desktop's codex_app tools, as magpie sends them on, carry a type on
// every object in their parameters, the root's and each nested one's: a
// relay serving tools without one had its whole turn turned away
// ("Mirasim: The request was rejected as invalid.", #2137). The check is
// the body the upstream received, so it fails when the fix isn't wired into
// the send path.
func TestToolObjectsTypedOnTheWire(t *testing.T) {
	for _, tc := range []struct {
		name string
		p    provider.Provider
	}{{
		// the failing route: Codex's Responses request relayed to a
		// provider that serves the model on Anthropic's Messages
		name: "translated to the Anthropic wire",
		p:    provider.Provider{ID: "relay", Name: "Relay", Key: "k", Models: []string{"m"}, Anthropic: "UP"},
	}, {
		name: "passed through on Responses",
		p:    provider.Provider{ID: "relay", Name: "Relay", Key: "k", Models: []string{"m"}, Responses: "UP"},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			testWire(t, tc.p)
		})
	}
}

func testWire(t *testing.T, p provider.Provider) {
	fresh(t)
	schema := `{"properties":{"a":{"type":"string"},"b":{"properties":{"c":{"type":"string"}},"required":["c"]}},"required":["a"]}`
	var mu sync.Mutex
	var bodies [][]byte
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, b)
		mu.Unlock()
		if strings.HasSuffix(r.URL.Path, "/messages") {
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, `event: message_start
data: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"usage":{"input_tokens":9,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"r1","object":"response","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":9,"output_tokens":1}}`)
	}))
	t.Cleanup(up.Close)
	if p.Anthropic != "" {
		p.Anthropic = up.URL + "/v1"
	}
	if p.Responses != "" {
		p.Responses = up.URL + "/v1"
	}
	if err := provider.Save(p); err != nil {
		t.Fatal(err)
	}
	s := New()
	body := `{"model":"relay/m","stream":false,"store":false,"tools":[{"type":"function","name":"t","parameters":` + schema + `}],` +
		`"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body)))
	if rec.Code != 200 {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != 1 {
		t.Fatalf("upstream asked %d times", len(bodies))
	}
	var q struct {
		Tools []struct {
			Parameters  map[string]any `json:"parameters"`
			InputSchema map[string]any `json:"input_schema"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(bodies[0], &q); err != nil || len(q.Tools) != 1 {
		t.Fatalf("%v %s", err, bodies[0])
	}
	params := q.Tools[0].Parameters
	if params == nil {
		params = q.Tools[0].InputSchema
	}
	if got := typelessObjects(params); len(got) != 0 {
		t.Fatalf("the upstream received untyped objects %v: %s", got, bodies[0])
	}
}

// typelessObjects lists the paths of nodes that declare properties or
// required but name no type.
func typelessObjects(v any) []string {
	var out []string
	var walk func(v any, path string)
	walk = func(v any, path string) {
		switch x := v.(type) {
		case map[string]any:
			if _, ok := x["type"]; !ok {
				if _, p := x["properties"]; p {
					out = append(out, path)
				} else if _, r := x["required"]; r {
					out = append(out, path)
				}
			}
			if props, ok := x["properties"].(map[string]any); ok {
				for k, sub := range props {
					walk(sub, path+"."+k)
				}
			}
			for _, k := range []string{"items", "additionalProperties"} {
				if sub, ok := x[k].(map[string]any); ok {
					walk(sub, path+"/"+k)
				}
			}
			for _, k := range []string{"anyOf", "oneOf", "allOf"} {
				if arr, ok := x[k].([]any); ok {
					for _, sub := range arr {
						walk(sub, path+"/"+k)
					}
				}
			}
		case []any:
			for _, sub := range x {
				walk(sub, path+"[]")
			}
		}
	}
	walk(v, "root")
	return out
}
