// ---- framebuffers ---------------------------------------------------------------------------------------------

void mtl_glGenFramebuffers(GLsizei n, GLuint *framebuffers) { gen_ids(s_next_framebuffer, n, framebuffers); REC(FN_glGenFramebuffers).i(n).i(n > 0 ? framebuffers[0] : 0); }
void mtl_glGenRenderbuffers(GLsizei n, GLuint *renderbuffers) { gen_ids(s_next_renderbuffer, n, renderbuffers); REC(FN_glGenRenderbuffers).i(n).i(n > 0 ? renderbuffers[0] : 0); }

void mtl_glDeleteFramebuffers(GLsizei n, const GLuint *framebuffers)
{
    for (GLsizei i = 0; i < n; i++) s_framebuffers.erase(framebuffers[i]);
}

void mtl_glDeleteRenderbuffers(GLsizei n, const GLuint *renderbuffers)
{
    for (GLsizei i = 0; i < n; i++) s_renderbuffers.erase(renderbuffers[i]);
}

void mtl_glBindFramebuffer(GLenum target, GLuint framebuffer)
{
    REC(FN_glBindFramebuffer).i(target).i(framebuffer);
    if (target == GL_READ_FRAMEBUFFER) return;
    s_fbo = framebuffer;
    if (framebuffer) s_framebuffers[framebuffer];
}

void mtl_glBindRenderbuffer(GLenum, GLuint renderbuffer) { REC(FN_glBindRenderbuffer).i(renderbuffer); s_rbo = renderbuffer; }

void mtl_glRenderbufferStorage(GLenum, GLenum internalformat, GLsizei width, GLsizei height)
{
    REC(FN_glRenderbufferStorage).i(internalformat).i(width).i(height);
    if (!s_rbo || width <= 0 || height <= 0) return;
    FormatInfo f = pick_format((GLint)internalformat, 0, 0);
    if (f.format != MTLPixelFormatDepth32Float) {
        complain("renderbuffer format %s is not supported", E(internalformat));
        return;
    }
    @autoreleasepool {
        Renderbuffer &rb = s_renderbuffers[s_rbo];
        rb.gl_internal = internalformat, rb.gl_w = width, rb.gl_h = height;
        rb.scale = scale_for(width, height);
        rb.mtl = new_depth_texture((NSUInteger)lround(width * rb.scale), (NSUInteger)lround(height * rb.scale),
                                   is_screen_sized(width, height) ? s_msaa : 1);
    }
}

void mtl_glFramebufferTexture2D(GLenum, GLenum attachment, GLenum, GLuint texture, GLint level)
{
    REC(FN_glFramebufferTexture2D).i(attachment).i(texture).i(level);
    Framebuffer *fb = find(s_framebuffers, s_fbo);
    if (!fb) return;
    if (level != 0) complain("rendering to mip level %d is not supported", level);
    if (attachment == GL_COLOR_ATTACHMENT0) fb->color_texture = texture;
    else if (attachment == GL_DEPTH_ATTACHMENT) fb->depth_texture = texture, fb->depth_renderbuffer = 0;
    else complain("framebuffer attachment %s is not supported", E(attachment));
}

void mtl_glFramebufferRenderbuffer(GLenum, GLenum attachment, GLenum, GLuint renderbuffer)
{
    REC(FN_glFramebufferRenderbuffer).i(attachment).i(renderbuffer);
    Framebuffer *fb = find(s_framebuffers, s_fbo);
    if (!fb) return;
    if (attachment == GL_DEPTH_ATTACHMENT) fb->depth_renderbuffer = renderbuffer, fb->depth_texture = 0;
    else complain("renderbuffer attachment %s is not supported", E(attachment));
}

// ---- shaders and programs ---------------------------------------------------------------------------------------

GLuint mtl_glCreateShader(GLenum type)
{
    GLuint id = ++s_next_object;
    s_shaders[id] = std::make_shared<Shader>();
    s_shaders[id]->type = type;
    REC(FN_glCreateShader).i(type).i(id);
    return id;
}

GLuint mtl_glCreateProgram(void)
{
    GLuint id = ++s_next_object;
    s_programs[id];
    REC(FN_glCreateProgram).i(id);
    return id;
}

void mtl_glShaderSource(GLuint shader, GLsizei count, const GLchar *const *string, const GLint *length)
{
    auto it = s_shaders.find(shader);
    if (it == s_shaders.end()) return;
    std::string &src = it->second->source;
    src.clear();
    for (GLsizei i = 0; i < count; i++) {
        if (!string[i]) continue;
        if (length && length[i] >= 0) src.append(string[i], (size_t)length[i]);
        else src.append(string[i]);
    }
    REC(FN_glShaderSource).i(shader).s(src);
}

// Translation needs both stages, so the work happens in glLinkProgram.
void mtl_glCompileShader(GLuint) {}

void mtl_glAttachShader(GLuint program, GLuint shader)
{
    REC(FN_glAttachShader).i(program).i(shader);
    Program *p = find(s_programs, program);
    auto it = s_shaders.find(shader);
    if (p && it != s_shaders.end()) p->shaders.push_back(it->second);
}

void mtl_glBindAttribLocation(GLuint program, GLuint index, const GLchar *name)
{
    REC(FN_glBindAttribLocation).i(program).i(index).s(name ? name : "");
    if (Program *p = find(s_programs, program); p && name) p->attribs[name] = (int)index;
}

void mtl_glLinkProgram(GLuint program)
{
    REC(FN_glLinkProgram).i(program);
    Program *p = find(s_programs, program);
    if (!p) return;
    @autoreleasepool {
        link_program(program, *p);
    }
}

void mtl_glDeleteShader(GLuint shader) { s_shaders.erase(shader); } // programs keep their own reference
void mtl_glDeleteProgram(GLuint program)
{
    s_programs.erase(program);
    s_last_key_program = 0;
}
void mtl_glUseProgram(GLuint program) { REC(FN_glUseProgram).i(program); s_program = program; }

// Compile and link status are always reported as success: a program that
// failed to translate is logged and its draws are skipped, which keeps the
// engine running where a reported failure would not.
void mtl_glGetShaderiv(GLuint, GLenum pname, GLint *params) { *params = pname == GL_COMPILE_STATUS ? GL_TRUE : 0; }
void mtl_glGetProgramiv(GLuint, GLenum pname, GLint *params) { *params = pname == GL_LINK_STATUS ? GL_TRUE : 0; }

void mtl_glGetShaderInfoLog(GLuint, GLsizei bufSize, GLsizei *length, GLchar *infoLog)
{
    if (length) *length = 0;
    if (bufSize > 0 && infoLog) infoLog[0] = 0;
}

void mtl_glGetProgramInfoLog(GLuint, GLsizei bufSize, GLsizei *length, GLchar *infoLog)
{
    if (length) *length = 0;
    if (bufSize > 0 && infoLog) infoLog[0] = 0;
}

GLint mtl_glGetUniformLocation(GLuint program, const GLchar *name)
{
    Program *p = find(s_programs, program);
    if (!p || !p->linked) return -1;
    auto it = p->location_by_name.find(name);
    if (it == p->location_by_name.end()) {
        size_t n = strlen(name);
        if (n > 3 && !strcmp(name + n - 3, "[0]")) it = p->location_by_name.find(std::string(name, n - 3));
    }
    return it == p->location_by_name.end() ? -1 : it->second;
}

void mtl_glUniform1f(GLint location, GLfloat v0) { REC(FN_glUniform1f).s(uniform_name(location)).f(v0); set_uniform(location, &v0, 1, 1, false); }
void mtl_glUniform1i(GLint location, GLint v0) { REC(FN_glUniform1i).s(uniform_name(location)).i(v0); set_uniform(location, &v0, 1, 1, true); }
void mtl_glUniform1iv(GLint location, GLsizei count, const GLint *value) { REC(FN_glUniform1iv).s(uniform_name(location)).i(count).blob(value, (size_t)std::max(count, 0) * 4); set_uniform(location, value, count, 1, true); }
void mtl_glUniform4fv(GLint location, GLsizei count, const GLfloat *value) { REC(FN_glUniform4fv).s(uniform_name(location)).i(count).blob(value, (size_t)std::max(count, 0) * 16); if (s_callsites) note_callsite("uniform", std::string(uniform_name(location)).c_str()); set_uniform(location, value, count, 4, false); }

void mtl_glUniformMatrix4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    REC(FN_glUniformMatrix4fv).s(uniform_name(location)).i(count).blob(value, (size_t)std::max(count, 0) * 64);
    if (transpose) complain("glUniformMatrix4fv with transpose is not supported");
    if (s_callsites) note_callsite("uniform", std::string(uniform_name(location)).c_str());
    set_uniform(location, value, count, 16, false);
}
