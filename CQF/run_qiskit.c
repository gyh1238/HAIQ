#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const char *py = "~/py311/bin/python";
    const char *script = "hardcoding_to_c.py";

    char cmd[2048];
    // stdout만 캡처 (stderr는 그대로 터미널로)
    snprintf(cmd, sizeof(cmd), "%s %s | python3 -m json.tool", py, script);

    FILE *fp = popen(cmd, "r");
    if (!fp) {
        perror("popen failed");
        return 1;
    }

    char buf[4096];
    while (fgets(buf, sizeof(buf), fp)) {
        fputs(buf, stdout);
    }

    int rc = pclose(fp);
    if (rc != 0) {
        fprintf(stderr, "Warning: python exited with code %d\n", rc);
    }
    return 0;
}

