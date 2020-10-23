#include "utils.hpp"



int main(void) {
  read_params();

  init_state_vectors();

  gen_conmat();

  integrate_brunel();

  delete_state_vectors();
  return 0;
}
