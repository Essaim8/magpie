package gateway

import (
	"bytes"
	"encoding/json"

	"github.com/yetone/magpie/internal/provider"
)

// A tool's parameters that declare properties or required but name no type
// are turned away by an upstream validating tools the way OpenAI's and
// Anthropic's APIs do, before any token is read: the whole turn fails
// though the tool is never called. Codex desktop's codex_app tools carry
// such schemas — zod's toJSONSchema leaves a z.object's properties and
// required without a type, at the root or on a nested field — and a relay
// in front of a strict model passes the refusal on (#2137: magpie's
// Mirasim provider answering "Mirasim: The request was rejected as
// invalid." for mirasim/gemini-4-argon through Codex, while the same model
// took the same tools in clients that carried their own).
//
// objectSchema (anthropic.go) folds a union at a tool's root and
// objectRootsBody (object_root.go) does the same for a Chat or Responses
// request, but both leave a schema that simply omits type as it came,
// guarding on the schema containing "Of". This gives every object that
// names no type its own, at the root and nested, on every wire; a schema
// that needs nothing is sent byte for byte, so a cache prefix doesn't
// move.

// objectTypes writes "type":"object" into a schema, and into every object
// nested under it, wherever a node declares properties or required but
// names no type, and reports whether anything changed. A node that states
// its shape another way — a union, a reference, a const or an enum — is
// left as it is: provider.ObjectRoot folds a union at a root and
// kimiToolEnumTypes gives Kimi's enum its type, neither wanting a type
// written here beside what it does.
func objectTypes(v any) bool {
	changed := false
	switch x := v.(type) {
	case map[string]any:
		if _, named := x["type"]; !named && !statesShapeOtherwise(x) {
			if _, p := x["properties"]; p {
				x["type"] = "object"
				changed = true
			} else if _, r := x["required"]; r {
				x["type"] = "object"
				changed = true
			}
		}
		if props, ok := x["properties"].(map[string]any); ok {
			for _, sub := range props {
				if objectTypes(sub) {
					changed = true
				}
			}
		}
		// the fields whose value is a schema of its own, at whatever depth
		for _, k := range []string{"items", "additionalProperties", "not", "if", "then", "else", "contains", "propertyNames", "unevaluatedProperties", "contentSchema"} {
			if sub, ok := x[k].(map[string]any); ok && objectTypes(sub) {
				changed = true
			}
		}
		for _, k := range []string{"anyOf", "oneOf", "allOf", "prefixItems"} {
			if arr, ok := x[k].([]any); ok {
				for _, sub := range arr {
					if objectTypes(sub) {
						changed = true
					}
				}
			}
		}
		// a named schema is folded into a union's branch by
		// provider.ObjectRoot, which may leave it here untyped, and a
		// request reaches it through a reference: it is typed too
		for _, k := range []string{"$defs", "definitions", "patternProperties", "dependentSchemas"} {
			if defs, ok := x[k].(map[string]any); ok {
				for _, sub := range defs {
					if objectTypes(sub) {
						changed = true
					}
				}
			}
		}
	case []any:
		for _, sub := range x {
			if objectTypes(sub) {
				changed = true
			}
		}
	}
	return changed
}

// statesShapeOtherwise is whether a node names its shape through a union, a
// reference, a const or an enum rather than a type.
func statesShapeOtherwise(m map[string]any) bool {
	for _, k := range []string{"anyOf", "oneOf", "allOf", "$ref", "const", "enum"} {
		if _, ok := m[k]; ok {
			return true
		}
	}
	return false
}

// needsObjectTypes is whether a schema could carry an object that names no
// type, a cheap guard so one without any is returned as it came.
func needsObjectTypes(schema []byte) bool {
	return bytes.Contains(schema, []byte(`"properties"`)) || bytes.Contains(schema, []byte(`"required"`))
}

// objectTypesBody is a body whose function tools' parameters have a type on
// every object, a namespace's included, and whether any was changed: a
// body with no tool schema, or none that needs a type, comes back as it
// was. Chat and Responses tools carry their schema in parameters, an
// Anthropic tool in input_schema.
func objectTypesBody(proto provider.Protocol, body []byte) ([]byte, bool) {
	if !needsObjectTypes(body) {
		return body, false
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	var m map[string]any
	if dec.Decode(&m) != nil || m == nil {
		return body, false
	}
	tools, _ := m["tools"].([]any)
	if !typeTools(proto, tools) {
		return body, false
	}
	out, err := marshalPlain(m)
	if err != nil {
		return body, false
	}
	return out, true
}

// typeTools writes the missing object types into the parameters of the
// function tools among tools, a Responses namespace's included, and reports
// whether it changed any.
func typeTools(proto provider.Protocol, tools []any) bool {
	changed := false
	for _, t := range tools {
		tm, _ := t.(map[string]any)
		if tm == nil {
			continue
		}
		if proto == provider.Responses && tm["type"] == "namespace" {
			nested, _ := tm["tools"].([]any)
			if typeTools(proto, nested) {
				changed = true
			}
			continue
		}
		fn := tm
		switch proto {
		case provider.Chat:
			fn, _ = tm["function"].(map[string]any)
		case provider.Anthropic:
			if ps, _ := tm["input_schema"].(map[string]any); ps != nil && objectTypes(ps) {
				changed = true
			}
			continue
		}
		if ps, _ := fn["parameters"].(map[string]any); ps != nil && objectTypes(ps) {
			changed = true
		}
	}
	return changed
}
