// ---- clears, viewport, fixed-function state ------------------------------------------------------

// A clear starts a new render pass whose load actions do the clearing; the
// draws that follow on the same target go into that pass.
void mtl_glClear(GLbitfield mask)
{
    REC(FN_glClear).i(mask);
    @autoreleasepool {
        Target t;
        if (resolve_target(t)) start_pass(t, mask);
    }
}

void mtl_glClearColor(GLclampf r, GLclampf g, GLclampf b, GLclampf a) { REC(FN_glClearColor).f(r).f(g).f(b).f(a); s_clear_color = MTLClearColorMake(r, g, b, a); }
void mtl_glViewport(GLint x, GLint y, GLsizei width, GLsizei height) { REC(FN_glViewport).i(x).i(y).i(width).i(height); s_viewport = {x, y, width, height}; }
void mtl_glDepthMask(GLboolean flag) { REC(FN_glDepthMask).i(flag); s_depth_mask = flag; }
void mtl_glDepthFunc(GLenum func) { REC(FN_glDepthFunc).i(func); s_depth_func = func; }
void mtl_glCullFace(GLenum mode) { REC(FN_glCullFace).i(mode); s_cull_face = mode; }
void mtl_glBlendEquation(GLenum mode) { REC(FN_glBlendEquation).i(mode); s_blend_eq = mode; }
void mtl_glPolygonMode(GLenum, GLenum mode) { REC(FN_glPolygonMode).i(mode); s_polygon_mode = mode; }

void mtl_glBlendFunc(GLenum sfactor, GLenum dfactor)
{
    REC(FN_glBlendFunc).i(sfactor).i(dfactor);
    s_blend_src = sfactor;
    s_blend_dst = dfactor;
}

static void set_capability(GLenum cap, bool on)
{
    switch (cap) {
    case GL_DEPTH_TEST: s_depth_test = on; break;
    case GL_CULL_FACE: s_cull = on; break;
    case GL_BLEND: s_blend = on; break;
    default: complain("glEnable/glDisable(%s) is ignored", E(cap));
    }
}

void mtl_glEnable(GLenum cap) { REC(FN_glEnable).i(cap); set_capability(cap, true); }
void mtl_glDisable(GLenum cap) { REC(FN_glDisable).i(cap); set_capability(cap, false); }

// ---- queries ------------------------------------------------------------------------------------------

void mtl_glGetIntegerv(GLenum pname, GLint *params)
{
    switch (pname) {
    case GL_MAX_TEXTURE_SIZE: *params = 16384; break;
    default:
        *params = 0;
        complain("unanswered glGetIntegerv %s", E(pname));
    }
}

void mtl_glGetFloatv(GLenum pname, GLfloat *params)
{
    switch (pname) {
    case GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT: *params = 16; break;
    default:
        *params = 0;
        complain("unanswered glGetFloatv %s", E(pname));
    }
}

// ---- buffers -------------------------------------------------------------------------------------------

void mtl_glGenBuffers(GLsizei n, GLuint *buffers) { gen_ids(s_next_buffer, n, buffers); REC(FN_glGenBuffers).i(n).i(n > 0 ? buffers[0] : 0); }

void mtl_glDeleteBuffers(GLsizei n, const GLuint *buffers)
{
    for (GLsizei i = 0; i < n; i++) s_buffers.erase(buffers[i]);
}

void mtl_glBindBuffer(GLenum target, GLuint buffer)
{
    REC(FN_glBindBuffer).i(target).i(buffer);
    if (target == GL_ARRAY_BUFFER) s_array_buffer = buffer;
    else if (target == GL_ELEMENT_ARRAY_BUFFER) s_element_buffer = buffer;
}

// Every glBufferData gets a fresh MTLBuffer, which is also what orphaning asks
// for: draws already encoded keep the old contents alive.
void mtl_glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum)
{
    REC(FN_glBufferData).i(target).i(size).blob(data, (size_t)std::max<GLsizeiptr>(size, 0));
    if (s_callsites) note_callsite("buffer filled", target == GL_ARRAY_BUFFER ? "(vertices)" : "(indices)");
    GLuint id = target == GL_ARRAY_BUFFER ? s_array_buffer : s_element_buffer;
    if (!id || size <= 0) return;
    @autoreleasepool {
        Buffer &b = s_buffers[id];
        b.mtl = data ? [s_device newBufferWithBytes:data length:(NSUInteger)size options:MTLResourceStorageModeShared]
                     : [s_device newBufferWithLength:(NSUInteger)size options:MTLResourceStorageModeShared];
    }
}

// ---- vertex arrays ---------------------------------------------------------------------------------------

void mtl_glEnableVertexAttribArray(GLuint index)
{
    REC(FN_glEnableVertexAttribArray).i(index);
    if (index < MAX_ATTRIBS) s_attribs[index].enabled = true;
}

void mtl_glDisableVertexAttribArray(GLuint index)
{
    REC(FN_glDisableVertexAttribArray).i(index);
    if (index < MAX_ATTRIBS) s_attribs[index].enabled = false;
}

void mtl_glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer)
{
    REC(FN_glVertexAttribPointer).i(index).i(size).i(type).i(normalized).i(stride).i((int64_t)(uintptr_t)pointer);
    if (index >= MAX_ATTRIBS) return;
    Attrib &a = s_attribs[index];
    a.buffer = s_array_buffer;
    a.size = size;
    a.type = type;
    a.normalized = normalized;
    a.stride = stride;
    a.offset = (uintptr_t)pointer;
}
